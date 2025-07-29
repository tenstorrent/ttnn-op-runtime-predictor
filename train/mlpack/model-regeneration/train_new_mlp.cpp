#include <chrono>
#include <cmath>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include <nlohmann/json.hpp>

#define MIN_NUM_ARGS 9
#define DEFAULT_MOMENTUM_DECAY 0.9
#define DEFAULT_SQUARED_GRADIENT_DECAY 0.999
#define DEFAULT_EPSILON 1e-8
#define DEFAULT_MAX_ITERATIONS 10000000
#define DEFAULT_TOLERANCE 1e-5

// todo: create helper file to contain common helper funcs across

mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>
initialize_model_architecture(int input_dim,
                              const std::vector<int> &hidden_layers) {
  mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> model;
  model.InputDimensions() =
      std::vector<size_t>({static_cast<unsigned long>(input_dim)});
  for (int i = 0; i < hidden_layers.size(); ++i) {
    model.Add<mlpack::Linear>(hidden_layers[i]);
    model.Add<mlpack::ReLU>();
  }
  model.Add<mlpack::Linear>(1);
  return model;
}

double R2Score(const arma::rowvec &actual, const arma::rowvec &predicted) {
  double ssRes = arma::accu(arma::square(actual - predicted));
  double ssTot = arma::accu(arma::square(actual - arma::mean(actual)));
  return 1 - (ssRes / ssTot);
}

auto load_and_split_data(const std::string &dataset_filepath,
                         const double valid_ratio = 0.2) {
  arma::mat trainX, validX, trainY, validY;
  mlpack::data::StandardScaler scaler;

  arma::mat dataset;
  if (!mlpack::data::Load(dataset_filepath, dataset, true)) {
    throw std::runtime_error("Error loading dataset csv!");
  }
  arma::mat features = dataset.rows(0, dataset.n_rows - 2);
  arma::mat labels = dataset.row(dataset.n_rows - 1);

  // print dataset size
  std::cout << "Dataset " << dataset_filepath << "loaded. Contains "
            << dataset.n_rows << " rows and " << dataset.n_cols << " columns."
            << std::endl;

  arma::mat scaled_features;
  scaler.Fit(features);
  scaler.Transform(features, scaled_features);

  arma::uvec indices = arma::randperm(scaled_features.n_cols);
  size_t valid_count = scaled_features.n_cols * valid_ratio;

  arma::uvec valid_indices = indices.head(valid_count);
  arma::uvec train_indices = indices.tail(scaled_features.n_cols - valid_count);

  trainX = scaled_features.cols(train_indices);
  trainY = labels.cols(train_indices);
  validX = scaled_features.cols(valid_indices);
  validY = labels.cols(valid_indices);

  return std::make_tuple(trainX, trainY, validX, validY, scaler);
}

ens::Adam create_optimizer(const nlohmann::json &optimizer_config) {
  return ens::Adam(
      optimizer_config["learning_rate"], optimizer_config["batch_size"],
      optimizer_config["momentum_decay"],
      optimizer_config["squared_gradient_decay"], optimizer_config["epsilon"],
      optimizer_config["max_iterations"], optimizer_config["tolerance"],
      true // shuffle data every epoch
  );
}

void evaluate_mlp(
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> &model,
    const arma::mat &X, const arma::mat &Y, const std::string &label) {
  arma::mat pred;
  model.Predict(X, pred);
  double r2 = R2Score(Y.row(0), pred.row(0));
  std::cout << label << " R² Score: " << r2 << std::endl;
}

double return_r2(
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> &model,
    const arma::mat &X, const arma::mat &Y, const std::string &label) {
  arma::mat pred;
  model.Predict(X, pred);
  double r2 = R2Score(Y.row(0), pred.row(0));
  std::cout << label << " R² Score: " << r2 << std::endl;
  return r2;
}

void save_model_and_scaler(
    const std::string &op_name,
    const mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>
        &model,
    const mlpack::data::StandardScaler &scaler) {
  std::string suffix = "_model.bin";
  std::string mlp_filename = op_name + suffix;
  mlpack::data::Save(mlp_filename, "model", model, true);
  mlpack::data::Save("exp_scaler.bin", "scaler", scaler, true);
  std::cout << "Model saved as " << mlp_filename << std::endl;
}

// todo
std::vector<int> parse_ints(char *arg) {

  if (arg == nullptr) {
    throw std::invalid_argument("Argument is null.");
  }
  std::vector<int> ints;
  std::istringstream ss(arg);
  std::string substring;
  while (std::getline(ss, substring, ',')) {
    ints.push_back(std::stoi(substring));
  }
  return ints;
}

auto parse_args(int argc, char **argv) {

  // required args
  std::string op_name, op_category, dataset;
  int input_dim;
  if (argc < MIN_NUM_ARGS) {
    throw std::invalid_argument("Not enough args.")
  }
  if (strcmp(argv[1], "--op-name") != 0) {
    throw std::invalid_argument(
        "Wrong first argument, should be --op-name <op name>");
  }
  if (strcmp(argv[3], "--op-category") != 0) {
    throw std::invalid_argument(
        "Wrong second argument, should be --op-category <op category>");
  }
  if (strcmp(argv[5], "--dataset") != 0) {
    throw std::invalid_argument(
        "Wrong third argument, should be --dataset <path/to/dataset.csv>");
  }
  if (strcmp(argv[7], "--input-dim") != 0) {
    throw std::invalid_argument(
        "Wrong fourth argument, should be --input-dim <mlp input dimension>");
  }

  // optional args
  std::optional<std::vector<int>> hidden_layers;
  std::optional<int> batch_size, max_iterations;
  std::optional<double> learning_rate, momentum_decay, squared_gradient_decay,
      epsilon, tolerance;
  for (int index = MIN_NUM_ARGS; index < argc; index += 2) {

    if (strcmp(argv[index], "--hidden-layers") == 0) {
      hidden_layers = parse_ints(argv[index + 1]);
    } else if (strcmp(argv[index], "--batch-size") == 0) {
      std::string arg = argv[index + 1];
      batch_size = std::stoi(arg);
    } else if (strcmp(argv[index], "--learning-rate") == 0) {
      std::string arg = argv[index + 1];
      learning_rate = std::stod(arg);
    } else if (strcmp(argv[index], "--momentum-decay") == 0) {
      std::string arg = argv[index + 1];
      momentum_decay = std::stod(arg);
    } else if (strcmp(argv[index], "--squared-gradient-decay") == 0) {
      std::string arg = argv[index + 1];
      squared_gradient_decay = std::stod(arg);
    } else if (strcmp(argv[index], "--epsilon") == 0) {
      std::string arg = argv[index + 1];
      epsilon = std::stod(arg);
    } else if (strcmp(argv[index], "--max-iterations") == 0) {
      std::string arg = argv[index + 1];
      max_iterations = std::stoi(arg);
    } else if (strcmp(argv[index], "--tolerance") == 0) {
      std::string arg = argv[index + 1];
      tolerance = std::stod(arg);
    } else {
      throw std::invalid_argument("Argument not recognized.");
    }
  }

  op_name = argv[2];
  op_category = argv[4];
  dataset = argv[6];

  std::string input_dim_arg = argv[8];
  input_dim = std::stoi(input_dim_arg);

  return std::make_tuple(op_name, op_category, dataset, input_dim,
                         hidden_layers, batch_size, learning_rate,
                         momentum_decay, squared_gradient_decay, epsilon,
                         max_iterations, tolerance);
}

nlohmann::json create_model_param_json(
    const std::string &op_name, const std::vector<int> &hidden_layers,
    const int &input_dim, const int &batch_size, const double &learning_rate,
    const double &momentum_decay, const double &squared_gradient_decay,
    const double &epsilon, const int &max_iterations, const double &tolerance) {
  nlohmann::json model_json;
  model_json["name"] = op_name;
  model_json["architecture_config"] = {{"hidden_layers", hidden_layers},
                                       {"input_size", input_dim}};
  model_json["optimizer_config"] = {
      {"batch_size", batch_size},
      {"learning_rate", learning_rate},
      {"momentum_decay", momentum_decay},
      {"squared_gradient_decay", squared_gradient_decay},
      {"epsilon", epsilon},
      {"max_iterations", max_iterations},
      {"tolerance", tolerance}};
  return model_json;
}

auto train_new_mlp(const std::string &op_name, const std::string &op_category,
                   const arma::mat &trainX, const arma::mat &trainY,
                   const arma::mat &validX, const arma::mat &validY,
                   const int &input_dim,
                   const std::optional<std::vector<int>> &hidden_layers_opt,
                   const std::optional<int> &batch_size_opt,
                   const std::optional<double> &learning_rate_opt,
                   const std::optional<double> &momentum_decay_opt,
                   const std::optional<double> &squared_gradient_decay_opt,
                   const std::optional<double> &epsilon_opt,
                   const std::optional<int> &max_iterations_opt,
                   const std::optional<double> &tolerance_opt) {

  const std::vector<std::vector<int>> hidden_layers_space =
      hidden_layers_opt ? std::vector<std::vector<int>>{*hidden_layers_opt}
                        : std::vector<std::vector<int>>{{128, 128, 128},
                                                        {256, 128, 128},
                                                        {128, 256, 128}};
  const std::vector<int> batch_size_space =
      batch_size_opt ? std::vector<int>{*batch_size_opt}
                     : std::vector<int>{32, 64, 128};
  const std::vector<double> learning_rate_space =
      learning_rate_opt ? std::vector<double>{*learning_rate_opt}
                        : std::vector<double>{0.01, 0.001, 0.0005};

  const double momentum_decay_val = momentum_decay_opt.has_value()
                                        ? momentum_decay_opt.value()
                                        : DEFAULT_MOMENTUM_DECAY;
  const double squared_gradient_decay_val =
      squared_gradient_decay_opt.has_value()
          ? squared_gradient_decay_opt.value()
          : DEFAULT_SQUARED_GRADIENT_DECAY;
  const double epsilon_val =
      epsilon_opt.has_value() ? epsilon_opt.value() : DEFAULT_EPSILON;
  const int max_iterations_val = max_iterations_opt.has_value()
                                     ? max_iterations_opt.value()
                                     : DEFAULT_MAX_ITERATIONS;
  const double tolerance_val =
      tolerance_opt.has_value() ? tolerance_opt.value() : DEFAULT_TOLERANCE;

  std::vector<double> r2_scores;
  std::vector<
      mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>>
      models;
  double best_valid_r2 = -1;
  mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>
      best_model;
  std::vector<nlohmann::json> model_params;
  nlohmann::json best_model_params;

  for (const std::vector<int> hidden_layers : hidden_layers_space) {
    for (const int batch_size : batch_size_space) {
      for (const double learning_rate : learning_rate_space) {

        nlohmann::json model_json = create_model_param_json(
            op_name, hidden_layers, input_dim, batch_size, learning_rate,
            momentum_decay_val, squared_gradient_decay_val, epsilon_val,
            max_iterations_val, tolerance_val);
        std::cout << "Training MLP with parameters: " << std::endl;
        std::cout << model_json.dump(4) << std::endl;

        auto model = initialize_model_architecture(input_dim, hidden_layers);
        mlpack::ens::Adam optimizer(learning_rate, batch_size, momentum_decay,
                                    squared_gradient_decay, epsilon,
                                    max_iterations, tolerance, true);

        auto start = std::chrono::high_resolution_clock::now();
        model.Train(trainX, trainY, optimizer);
        auto end = std::chrono::high_resolution_clock::now();

        double total_time = duration_cast<milliseconds>(end - start).count();
        std::cout << "Training completed in " << total_time / 1000.0
                  << " seconds.\n";

        evaluate_mlp(model, trainX, trainY, "Training");
        double valid_r2 = return_r2(model, validX, validY, "Validation");

        valid_r2_scores.push_back(valid_r2);
        models.push_back(model);
        model_params.push_back(model_json);

        if (valid_r2 > best_valid_r2) {
          best_valid_r2 = valid_r2;
          best_model = model;
          best_model_params = model_json;
        }
      }
    }
  }
  return std::make_tuple(best_model, best_valid_r2, best_model_params);
}

int main(int argc, char **argv) {
  try {
    auto [op_name, op_category, dataset_filepath, input_dim, hidden_layers,
          batch_size, learning_rate, momentum_decay, squared_gradient_decay,
          epsilon, max_iterations, tolerance] = parse_args(argc, argv);

    auto [trainX, trainY, validX, validY, scaler] =
        load_and_split_data(dataset_filepath);

    auto [model, r2, params] = train_new_mlp(
        op_name, op_category, trainX, trainY, input_dim, hidden_layers,
        batch_size, learning_rate, momentum_decay, squared_gradient_decay,
        epsilon, max_iterations, tolerance);

    const std::string &config_filepath = "mlp_config.json";
    save_mlp_config(params, op_name, config_filepath);
    save_model_and_scaler(op_name, model, scaler);
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return -1;
  }
  return 0;
}