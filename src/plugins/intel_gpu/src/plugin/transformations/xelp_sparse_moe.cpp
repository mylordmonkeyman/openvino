// Copyright (C) 2026 OpenVINO Xe-LP contributors
// SPDX-License-Identifier: Apache-2.0
#include "xelp_sparse_moe.hpp"

#include <iostream>
#include <memory>
#include <vector>

#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/pp.hpp"
#include "ov_ops/gather_compressed.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"

namespace ov::intel_gpu {
namespace {
namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;

std::shared_ptr<v0::Constant> shape_vec(const ov::Shape& dims) {
    std::vector<int64_t> data(dims.begin(), dims.end());
    return v0::Constant::create(ov::element::i64, ov::Shape{data.size()}, data);
}

std::shared_ptr<v1::Reshape> reshape(const ov::Output<ov::Node>& data, const ov::Shape& target, bool zero = false) {
    return std::make_shared<v1::Reshape>(data, shape_vec(target), zero);
}
}  // namespace

XelpGatherMatmulCompressedToGather::XelpGatherMatmulCompressedToGather() {
    using namespace ov::pass::pattern;
    using ov::op::internal::GatherMatmulCompressed;
    auto root = wrap_type<GatherMatmulCompressed>();
    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        auto bgm = ov::as_type_ptr<GatherMatmulCompressed>(m.get_match_root());
        if (!bgm || transformation_callback(bgm)) return false;
        if (bgm->get_input_element_type(1) != ov::element::u4 &&
            bgm->get_input_element_type(1) != ov::element::i4) return false;

        const auto b = bgm->input_value(1);
        const auto scales = bgm->input_value(4);
        const auto zp = bgm->input_value(5);
        // ConvertGatherMatmulToGatherMatmulCompressed flattens groupwise
        // u4 weights to [experts, N, K] and quant params to [experts, N, groups].
        const auto bshape = b.get_shape();
        const auto sshape = scales.get_shape();
        const auto zshape = zp.get_shape();
        if (bshape.size() != 3 || sshape.size() != 3 || zshape.size() != 3) return false;
        const size_t experts = bshape[0], n = bshape[1], k = bshape[2], groups = sshape[2];
        if (experts != 256 || groups == 0 || k % groups != 0 || k / groups != 128) return false;
        if (sshape != zshape || sshape[0] != experts || sshape[1] != n) return false;
        const auto inds = bgm->input_value(2);
        const auto activation = bgm->input_value(0);
        if (inds.get_partial_shape().rank().is_dynamic() || inds.get_partial_shape().rank().get_length() != 2)
            return false;
        if (activation.get_partial_shape().rank().is_dynamic() ||
            activation.get_partial_shape().rank().get_length() != 3) return false;

        // Avoid a dense all-expert dequantization. GatherCompressed extracts
        // only the runtime topK experts and dequantizes those selected slices.
        const size_t group_size = k / groups;
        auto packed = reshape(b, {experts, n * groups, group_size});
        auto scale3 = reshape(scales, {experts, n * groups, 1});
        auto zp3 = reshape(zp, {experts, n * groups, 1});
        auto axis = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        auto gathered = std::make_shared<ov::op::internal::GatherCompressed>(
            packed, inds, axis, 0, scale3, zp3);

        // B: [tokens, topK, N*groups, group_size] -> [tokens, topK, N, K].
        auto weights = reshape(gathered, {0, 0, n, k}, true);
        auto decoded = std::make_shared<v0::Convert>(weights, activation.get_element_type());

        // A: [1 or topK, tokens, K] -> [tokens, 1 or topK, 1, K].
        auto trans_a = std::make_shared<v1::Transpose>(
            activation, v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 0, 2}));
        auto expanded_a = std::make_shared<v0::Unsqueeze>(
            trans_a, v0::Constant::create(ov::element::i32, ov::Shape{1}, {2}));
        auto matmul = std::make_shared<v0::MatMul>(expanded_a, decoded, false, true);
        // [tokens, topK, 1, N] -> [topK, tokens, N].
        auto squeezed = std::make_shared<v0::Squeeze>(
            matmul, v0::Constant::create(ov::element::i32, ov::Shape{1}, {2}));
        std::shared_ptr<ov::Node> output = std::make_shared<v1::Transpose>(
            squeezed, v0::Constant::create(ov::element::i32, ov::Shape{3}, {1, 0, 2}));
        if (ov::shape_size(bgm->get_input_shape(3)) > 0)
            output = std::make_shared<v1::Add>(output, bgm->input_value(3));
        output->set_friendly_name(bgm->get_friendly_name());
        ov::copy_runtime_info(bgm, output);
        std::cerr << "XELP_SPARSE_MOE_REPLACED " << bgm->get_friendly_name()
                  << " (256 experts -> dynamic topK GatherCompressed + MatMul)\n";
        ov::replace_node(bgm, output);
        return true;
    };
    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(
        root, "XelpGatherMatmulCompressedToGather");
    register_matcher(matcher, callback);
}
}  // namespace ov::intel_gpu
