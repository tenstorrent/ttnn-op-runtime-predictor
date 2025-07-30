#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include <nlohmann/json.hpp>

#include "mlp_config_utils.hpp"
#include "train_mlp_utils.hpp"

#define NUM_ARGS 5

using namespace std::chrono;

auto parse_args(int argc, char **argv) {
  if (argc < NUM_ARGS) {
    throw std::invalid_argument("Not enough args");
  }
  if (strcmp(argv[1], "--op-name") != 0) {
    throw std::invalid_argument(
        "Wrong first argument, should be --op-name <op name>");
  }
  if (strcmp(argv[3], "--dataset") != 0) {
    throw std::invalid_argument(
        "Wrong second argument, should be --dataset <path/to/dataset.csv>");
  }
  std::string op_name = argv[2];
  std::string dataset_filepath = argv[4];
  return std::make_tuple(op_name, dataset_filepath);
}


int main(int argc, char **argv) {
  try {
    auto [op_name, dataset_filepath] = parse_args(argc, argv);

    auto [trainX, trainY, validX, validY, scaler] =
        load_and_split_data(dataset_filepath);

    const std::string config_filepath = "mlp_config.json";
    const nlohmann::json mlp_config = load_mlp_config(op_name, config_filepath);

    auto model =
        initialize_model_architecture(mlp_config["architecture_config"]);
    auto optimizer = create_optimizer(mlp_config["optimizer_config"]);

    std::cout << "Training begun." << std::endl;
    auto start = high_resolution_clock::now();
    model.Train(trainX, trainY, optimizer);
    auto end = high_resolution_clock::now();

    double total_time = duration_cast<milliseconds>(end - start).count();
    std::cout << "Training completed in " << total_time / 1000.0
              << " seconds.\n";

    evaluate_mlp(model, trainX, trainY, "Training");
    evaluate_mlp(model, validX, validY, "Validation");

    save_model_and_scaler(op_name, model, scaler);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}