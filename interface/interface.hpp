#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>

#include "ops/include/ops.hpp"

template<typename ... Args>
uint64_t get_runtime_from_model(const std::string& op_name, Args&& ... args){

    if (op_name == "ttnn::exp") {
        //return predict_exp_runtime(std::forward<Args>(args)...);
        return 1; //for test purposes
    } else {
        return 0;
    }

}
