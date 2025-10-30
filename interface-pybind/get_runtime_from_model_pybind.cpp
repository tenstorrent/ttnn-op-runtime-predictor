// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "pybind11_json/pybind11_json.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "interface/interface.hpp"

namespace py = pybind11;

// python module ttnn_op_runtime_predictor. This python module exposes the
// function get_runtime_from_model (from interface/interface.hpp) to python.
// There are overloads for each template instatiation of get_runtime_from_model.
// This is to allow the pybinded function to retain the built-in dispatch
// capability of the C++ function.

PYBIND11_MODULE(ttnn_op_runtime_predictor, m) {
  m.doc() = "Python bindings for the TTNN Op Runtime Predictor";

  m.def("get_runtime_from_model",
        &op_perf::get_runtime_from_model<const nlohmann::json &,
                                         const nlohmann::json &>,
        "Query a model for the runtime prediction of an operation."
        "This overloaded function can be used for any ttnn eltwise unary op "
        "which has a trained model on the repo, or "
        "ttnn.transformer.concatenate_heads op.",
        py::arg("op_name"), py::arg("tensor_json"),
        py::arg("output_memory_config") = nlohmann::json::object());

  m.def("get_runtime_from_model",
        &op_perf::get_runtime_from_model<const nlohmann::json &, const int &,
                                         const std::optional<int> &,
                                         const bool &>,
        "Query a model for the runtime prediction of an operation."
        "This overloaded function can be used for "
        "ttnn.experimental.create_qkv_heads op.",
        py::arg("op_name") = "create_qkv_heads", py::arg("tensor_json"),
        py::arg("num_heads"), py::arg("num_kv_heads") = std::nullopt,
        py::arg("transpose_k_heads") = false);

  m.def("get_runtime_from_model",
        &op_perf::get_runtime_from_model<
            const nlohmann::json &, const nlohmann::json &,
            const nlohmann::json &, const nlohmann::json &,
            const std::optional<nlohmann::json> &,
            const std::optional<nlohmann::json> &, const bool &, const float &,
            int &, int &, const nlohmann::json &, const int &, int &, int &,
            int &, int &, const bool &, const bool &>,
        "Query a model for the runtime prediction of an operation."
        "This overloaded function can be used for "
        "ttnn.transformer.paged_scaled_dot_product_attention_decode op.",
        py::arg("op_name") = "paged_scaled_dot_product_attention_decode",
        py::arg("q_tensor_json"), py::arg("k_tensor_json"),
        py::arg("v_tensor_json"), py::arg("page_table_tensor_json"),
        py::arg("optional_cur_pos_tensor_json") = std::nullopt,
        py::arg("optional_attn_mask_tensor_json") = std::nullopt,
        py::arg("is_causal") = false, py::arg("optional_scale") = 1.0f,
        py::arg("k_chunk_size") = 0, py::arg("q_chunk_size") = 0,
        py::arg("output_memory_config"), py::arg("math_fidelity") = 0,
        py::arg("math_approx_mode") = 0, py::arg("fp32_dest_acc_en") = 0,
        py::arg("packer_l1_acc") = 0, py::arg("exp_approx_mode") = 0,
        py::arg("use_sdpa_program_config") = true,
        py::arg("use_compute_kernel_config") = true);
}
