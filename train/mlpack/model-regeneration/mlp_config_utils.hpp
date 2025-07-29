#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>

// save / update mlp's saved architecture and training parameters in config file
void save_mlp_config(const nlohmann::json &model_json,
                     const std::string &model_name,
                     const std::string &config_path) {
  nlohmann::json config;
  std::ifstream infile(config_path);
  if (!infile) {
    throw std::runtime_error("Error: could not open config file: " +
                             config_path);
  }
  infile >> config;
  infile.close();

  config["models"][model_name] = model_json;

  std::ofstream outfile(config_path);
  if (outfile) {
    outfile << config.dump(4);
    outfile.close();
    std::cout << "Config for model '" << model_name << "' saved to "
              << config_path << std::endl;
  } else {
    throw std::runtime_error("Error: could not write to config file: " +
                             config_path);
  }
}

// load mlp's architecture and training parameters as a json object from config
// file
nlohmann::json load_mlp_config(const std::string &model_name,
                               const std::string &config_path) {
  nlohmann::json config;
  std::ifstream infile(config_path);
  if (!infile) {
    throw std::runtime_error("Error: could not open config file: " +
                             config_path);
  }
  infile >> config;
  infile.close();
  if (config.contains("models") && config["models"].contains(model_name)) {
    // print full json object
    std::cout << "Loaded config for model '" << model_name << "':\n"
              << config["models"][model_name].dump(4) << std::endl;
    return config["models"][model_name];
  } else {
    throw std::runtime_error("Error: model '" + model_name +
                             "' not found in config file: " + config_path);
  }
}