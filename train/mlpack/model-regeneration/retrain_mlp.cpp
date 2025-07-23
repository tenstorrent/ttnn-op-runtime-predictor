#include <chrono>
#include <iostream>
#include <string>
#include <cmath>

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>

#include <nlohmann/json.hpp>

#include "mlp_config_utils.hpp"

#define NUM_ARGS 5

double R2ScoreA(const arma::rowvec& actual, const arma::rowvec& predicted)
{
    double ssRes = arma::accu(arma::square(actual - predicted));
    double ssTot = arma::accu(arma::square(actual - arma::mean(actual)));
    return 1 - (ssRes / ssTot);
}

mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>
initialize_model_architecture(const nlohmann::json architecture_config_json) {

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

int main(int argc, char** argv){
    if(argc < NUM_ARGS){
        std::cerr << "Not enough args" << std::endl;
        return -1;
    }
    if(strcmp(argv[1], "--op-name") != 0){
        std::cerr << "Wrong first argument, should be --op-name <op name>" << std::endl;
        return -1;
    }
    if(strcmp(argv[3], "--dataset") != 0){
        std::cerr << "Wrong second argument, should be --dataset <path/to/dataset.csv>" << std::endl;
        return -1;
    }
    std::string op_name(argv[2]);
    std::string dataset_filepath(arg[4]);

    //todo: optional check if op_name is a valid ttnn op?
    //can check config file to see if that op name exists...

    arma::mat dataset;
    if (!data::Load(dataset_filepath, dataset, true)){ 
        std::cerr << "Error loading dataset csv!" << std::endl;
        return -1;
    }

    arma::mat features = dataset.rows(0, dataset.n_rows - 2);  
    arma::mat labels = dataset.row(dataset.n_rows - 1);

    //scale training data
    mlpack::data::StandardScaler scaler;
    arma::mat scaled_features;
    scaler.Fit(features);
    scaler.Transform(features, scaled_features);

    // Split into train/validation (e.g., 80/20 split)
    arma::mat trainX, validX, trainY, validY;
    const double valid_ratio = 0.2;
    arma::uvec indices = arma::randperm(scaled_features.n_cols);
    size_t valid_count = scaled_features.n_cols * valid_ratio;

    arma::uvec valid_indices = indices.head(valid_count);
    arma::uvec train_indices = indices.tail(scaled_features.n_cols - valid_count);

    trainX = scaled_features.cols(train_indices);
    trainY = labels.cols(train_indices);
    validX = scaled_features.cols(valid_indices);
    validY = labels.cols(valid_indices);
    
    //load architecture and hyperparameters from config file. todo: decide location of config file
    const std::string& config_filepath = "path/to/config.json";
    const nlohmann::json mlp_config = load_mlp_config(op_name, config_filepath);
    //todo: error check above function

    
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> model = initialize_model_architecture(mlp_config["architecture_config"]);

    nlohmann::json optimizer_config = mlp_config["optimizer_config"];
    //todo: do I need mlpack:: here, because I am not   using namespace mlpack;?
    mlpack::ens::Adam optimizer(
                            optimizer_config["learning_rate"],  // Learning rate
                            optimizer_config["batch_size"],    // Batch size
                            optimizer_config["momentum_decay"],   // Momentum decay
                            optimizer_config["squared_gradient_decay"], // Squared gradient decay
                            optimizer_config["epsilon"],  // Small epsilon to avoid division by zero
                            optimizer_config["max_iterations"], // Max iterations
                            optimizer_config["tolerance"],  // Tolerance (stopping criterion)
                            true // Shuffle data every epoch
    ); 

    auto start = std::chrono::high_resolution_clock::now();
    model.Train(trainX, trainY, optimizer);
    auto end = std::chrono::high_resolution_clock::now();

    double total_time = std::chrono::duration_cast<milliseconds>(end - start).count();
    std::cout << "Training completed in " << total_time / 1000.0 << " seconds.\n";

    //predict on training set
    arma::mat trainPred;
    model.Predict(trainX, trainPred);
    double trainR2 = R2ScoreA(trainY.row(0), trainPred.row(0));
    std::cout << "Training R² Score: " << trainR2 << std::endl;

    //predict on validation set
    arma::mat validPred;
    model.Predict(validX, validPred);
    double validR2 = R2ScoreA(validY.row(0), validPred.row(0));
    std::cout << "Validation R² Score: " << validR2 << std::endl;

    //save model and scaler
    std::string suffix = "_model.bin";
    std::string mlp_filename = op_name + suffix;
    mlpack::data::Save(mlp_filename, "model", model, true);
    mlpack::data::Save("exp_scaler.bin", "scaler", scaler, true);
    std::cout << "Model saved as " << mlp_filename << std::endl;

}