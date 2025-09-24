#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "pybind11_json/pybind11_json.hpp"

#include "interface/interface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(ttnn_op_runtime_predictor, m) {
    m.doc() = "Python bindings for the TTNN Op Runtime Predictor";

    m.def("get_exp_runtime", &op_perf::get_runtime_from_model<const nlohmann::json&, const nlohmann::json&>,
          "Get the exp runtime prediction from the model",
          py::arg("op_name") = "exp", py::arg("tensor_json"), py::arg("output_memory_config") = nlohmann::json::object());

      m.def("get_concatenate_heads_runtime", &op_perf::get_runtime_from_model<const nlohmann::json&, const nlohmann::json&>,
          "Get the concatenate_heads runtime prediction from the model",
          py::arg("op_name") = "concatenate_heads", py::arg("tensor_json"), py::arg("output_memory_config") = nlohmann::json::object());

    m.def("get_create_qkv_heads_runtime", &op_perf::get_runtime_from_model<const nlohmann::json&, const int&, const std::optional<int>&, const bool&>,
          "Get the create_qkv_heads runtime prediction from the model",
          py::arg("op_name") = "create_qkv_heads", py::arg("tensor_json"), py::arg("num_heads"), py::arg("num_kv_heads") = std::nullopt, py::arg("transpose_k_heads") = false);
}
