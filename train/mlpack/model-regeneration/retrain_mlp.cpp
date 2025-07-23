#include <chrono>
#include <iostream>
#include <string>
#include <cmath>
#include <stdexcept>

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include <nlohmann/json.hpp>

#include "mlp_config_utils.hpp"

#define NUM_ARGS 5

using namespace std::chrono;

double R2ScoreA(const arma::rowvec& actual, const arma::rowvec& predicted)
{
    double ssRes = arma::accu(arma::square(actual - predicted));
    double ssTot = arma::accu(arma::square(actual - arma::mean(actual)));
    return 1 - (ssRes / ssTot);
}

mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> initialize_model_architecture(const nlohmann::json& architecture_config){
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> model;
    model.InputDimensions() = std::vector<size_t>({static_cast<unsigned long>(architecture_config["input_size"])});
    int num_hidden_layers = architecture_config["hidden_layers"].size();
    std::vector<int> hidden_layers = architecture_config["hidden_layers"];
    for (int i = 0; i < num_hidden_layers; ++i) {
        model.Add<mlpack::Linear>(hidden_layers[i]);
        model.Add<mlpack::ReLU>();
    }
    model.Add<mlpack::Linear>(1);
    return model;
}

void parse_args(int argc, char** argv, std::string& op_name, std::string& dataset_filepath){
    if(argc < NUM_ARGS){
        throw std::invalid_argument("Not enough args");
    }
    if(strcmp(argv[1], "--op-name") != 0){
        throw std::invalid_argument("Wrong first argument, should be --op-name <op name>");
    }
    if(strcmp(argv[3], "--dataset") != 0){
        throw std::invalid_argument("Wrong second argument, should be --dataset <path/to/dataset.csv>");
    }
    op_name = argv[2];
    dataset_filepath = argv[4];
}

void load_and_split_data(const std::string& dataset_filepath,arma::mat& trainX, arma::mat& trainY,arma::mat& validX, arma::mat& validY,mlpack::data::StandardScaler& scaler,double valid_ratio = 0.2){
    arma::mat dataset;
    if (!mlpack::data::Load(dataset_filepath, dataset, true)){
        throw std::runtime_error("Error loading dataset csv!");
    }
    arma::mat features = dataset.rows(0, dataset.n_rows - 2);
    arma::mat labels = dataset.row(dataset.n_rows - 1);

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
}

mlpack::ens::Adam create_optimizer(const nlohmann::json& optimizer_config){
    return mlpack::ens::Adam(
        optimizer_config["learning_rate"],
        optimizer_config["batch_size"],
        optimizer_config["momentum_decay"],
        optimizer_config["squared_gradient_decay"],
        optimizer_config["epsilon"],
        optimizer_config["max_iterations"],
        optimizer_config["tolerance"],
        true //shuffle data every epoch
    );
}

void evaluate_and_print(mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>& model,const arma::mat& X, const arma::mat& Y, const std::string& label){
    arma::mat pred;
    model.Predict(X, pred);
    double r2 = R2ScoreA(Y.row(0), pred.row(0));
    std::cout << label << " R² Score: " << r2 << std::endl;
}

void save_model_and_scaler(const std::string& op_name, mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>& model, mlpack::data::StandardScaler& scaler){
    std::string suffix = "_model.bin";
    std::string mlp_filename = op_name + suffix;
    mlpack::data::Save(mlp_filename, "model", model, true);
    mlpack::data::Save("exp_scaler.bin", "scaler", scaler, true);
    std::cout << "Model saved as " << mlp_filename << std::endl;
}

int main(int argc, char** argv){
    try {
        std::string op_name, dataset_filepath;
        parse_args(argc, argv, op_name, dataset_filepath);

        arma::mat trainX, validX, trainY, validY;
        mlpack::data::StandardScaler scaler;
        load_and_split_data(dataset_filepath, trainX, trainY, validX, validY, scaler);

        // Load architecture and hyperparameters from config file
        const std::string config_filepath = "path/to/config.json";
        const nlohmann::json mlp_config = load_mlp_config(op_name, config_filepath);

        auto model = initialize_model_architecture(mlp_config["architecture_config"]);
        auto optimizer = create_optimizer(mlp_config["optimizer_config"]);

        auto start = high_resolution_clock::now();
        model.Train(trainX, trainY, optimizer);
        auto end = high_resolution_clock::now();

        double total_time = duration_cast<milliseconds>(end - start).count();
        std::cout << "Training completed in " << total_time / 1000.0 << " seconds.\n";

        evaluate_and_print(model, trainX, trainY, "Training");
        evaluate_and_print(model, validX, validY, "Validation");

        save_model_and_scaler(op_name, model, scaler);
    }
    catch(const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}