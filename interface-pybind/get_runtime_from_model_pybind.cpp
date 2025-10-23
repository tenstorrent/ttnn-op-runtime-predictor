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
      //add def for paged sdpa decode        
}
