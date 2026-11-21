// Copyright (C) 2026 OpenVINO Xe-LP contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ov::intel_gpu {

// An opt-in Xe-LP fallback for MoE compressed expert GEMM that uses existing
// compressed Gather and conventional MatMul implementations.
class XelpGatherMatmulCompressedToGather : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("XelpGatherMatmulCompressedToGather");
    XelpGatherMatmulCompressedToGather();
};
}  // namespace ov::intel_gpu
