"""Source-plugin smoke: original Qwen3.6 layer-0 MoE with real TopK router."""
import os
import time
from pathlib import Path

import numpy as np
import openvino as ov
from openvino import opset13 as ops

ROOT = Path(r"C:\ai\openvino-xelp")
MODEL = Path(r"C:\ai\models\OpenVINO\Qwen3.6-35B-A3B-int4-ov\openvino_language_model.xml")
DLL = ROOT / "bin/intel64/Release/openvino_intel_gpu_plugin.dll"
assert os.environ.get("XELP_SPARSE_INT4_MOE") == "1", "Xe-LP sparse opt-in is required"
os.environ.pop("XELP_SKIP_CONSTPROP", None)
os.environ["PATH"] = str(DLL.parent) + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(str(DLL.parent))

core = ov.Core()
core.register_plugin(str(DLL), "XELP_GPU")
print("GPU:", core.get_property("XELP_GPU", "FULL_DEVICE_NAME"), flush=True)
model = core.read_model(str(MODEL))
nodes = list(model.get_ops())

topks = [n for n in nodes if n.get_type_name() == "TopK" and
         "layers.0.mlp.gate/" in n.get_friendly_name()]
assert len(topks) == 1, f"TopK candidates: {len(topks)}"
router_softmax = topks[0].input_value(0).get_node()
assert router_softmax.get_type_name() == "Softmax"
router_mm = router_softmax.input_value(0).get_node()
assert router_mm.get_type_name() == "MatMul"

roots = []
for n in nodes:
    if n.get_type_name() != "ReduceSum" or "layers.0.mlp" not in n.get_friendly_name():
        continue
    shape = n.get_output_partial_shape(0)
    if shape.rank().is_static() and len(shape) and shape[-1].is_static() and shape[-1].get_length() == 2048:
        roots.append(n)
print("FINAL_AGGREGATION_CANDIDATES:", [n.get_friendly_name() for n in roots], flush=True)
assert len(roots) == 1, "Could not isolate final MoE sum"
tiles = [n for n in nodes if n.get_type_name() == "Tile" and "layers.0.mlp" in n.get_friendly_name()]
print("TILE_CANDIDATES:", [n.get_friendly_name() for n in tiles], flush=True)
assert len(tiles) == 1, "Could not isolate tiled MoE input"

x = ops.parameter([1, 2048], dtype=ov.Type.f32)
x.set_friendly_name("xelp_original_layer0_moe_input")
tiles[0].input(0).replace_source_output(x.output(0))
router_mm.input(0).replace_source_output(x.output(0))
probe = ov.Model([roots[0].output(0)], [x], "xelp_original_layer0_moe")
nops = len(probe.get_ops())
print("ORIGINAL_MOE_GRAPH_NODES:", nops,
      "OUTPUT:", probe.output(0).get_partial_shape(), flush=True)
assert nops < 600, "Subgraph still includes upstream model; refusing large compile"
assert len(probe.get_parameters()) == 1, "Unexpected upstream graph parameters"

mib = 1048576
def memory(stage):
    d = core.get_property("XELP_GPU", "GPU_MEMORY_STATISTICS")
    print("GPU_MEMORY", stage, {k: round(v/mib, 2) for k,v in d.items()},
          "TOTAL_MIB", round(sum(d.values())/mib, 2), flush=True)
    return d

memory("BEFORE_COMPILE")
t = time.perf_counter()
gpu = core.compile_model(probe, "XELP_GPU")
print("GPU_COMPILE_SUCCESS_SECONDS", round(time.perf_counter()-t, 3), flush=True)
memory("AFTER_COMPILE")
rt_names = [n.get_friendly_name() for n in gpu.get_runtime_model().get_ops()]
print("GATHER_NODES", [s for s in rt_names if "gather" in s.lower()][:20], flush=True)
hidden = (np.random.default_rng(2026).standard_normal((1, 2048)) * 0.3).astype(np.float32)
t = time.perf_counter()
out = np.asarray(next(iter(gpu.infer_new_request([hidden]).values()))).copy().astype(np.float32)
print("GPU_INFER_SECONDS", round(time.perf_counter()-t, 4),
      "SHAPE", out.shape, "FINITE", np.isfinite(out).all(), flush=True)
assert np.isfinite(out).all()

t = time.perf_counter()
cpu = core.compile_model(probe, "CPU")
ref = np.asarray(next(iter(cpu.infer_new_request([hidden]).values()))).copy().astype(np.float32)
print("CPU_COMPILE_AND_INFER_SECONDS", round(time.perf_counter()-t, 3), flush=True)
assert out.shape == ref.shape, "Output shape mismatch"
maxerr = float(np.max(np.abs(out-ref)))
rell2 = float(np.linalg.norm((out-ref).astype(np.float64).ravel())/
              max(np.linalg.norm(ref.astype(np.float64).ravel()),1e-20))
print("CPU_GPU_CHECK max_abs",maxerr,"relative_l2",rell2,flush=True)
assert rell2 < 0.04 and np.allclose(out,ref,rtol=0.03,atol=0.0007), "GPU/CPU mismatch"
print("SOURCE_MOE_SMOKE_SUCCESS",flush=True)
