#include <iostream>
#include <mlpack.hpp>
#include <nlohmann/json.hpp>

#include "interface/include/interface.hpp"

int main(){
    std::cout << "compile main test" << std::endl;
    int i = get_runtime_from_model("ttnn::exp");
    std::cout << i << std::endl;
}