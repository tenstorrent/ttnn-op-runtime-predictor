// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <mlpack.hpp>
#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <armadillo>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <stdexcept>

//r2 score calculation
double R2Score(const arma::rowvec &actual, const arma::rowvec &predicted) {
    double ssRes = arma::accu(arma::square(actual - predicted));
    double ssTot = arma::accu(arma::square(actual - arma::mean(actual)));
    return 1 - (ssRes / ssTot);
}

//model architecture initialization (from input_dim and hidden_layers)
mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>
initialize_model_architecture(int input_dim, const std::vector<int> &hidden_layers) {
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> model;
    model.InputDimensions() = std::vector<size_t>({static_cast<unsigned long>(input_dim)});
    for (int i = 0; i < hidden_layers.size(); ++i) {
        model.Add<mlpack::Linear>(hidden_layers[i]);
        model.Add<mlpack::ReLU>();
    }
    model.Add<mlpack::Linear>(1);
    return model;
}

//model architecture initialization (from JSON config)
mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization>
initialize_model_architecture(const nlohmann::json &architecture_config) {
    int input_dim = architecture_config["input_size"];
    std::vector<int> hidden_layers = architecture_config["hidden_layers"];
    return initialize_model_architecture(input_dim, hidden_layers);
}

//load and split data
auto load_and_split_data(const std::string &dataset_filepath, const double valid_ratio = 0.2) {
    arma::mat trainX, validX, trainY, validY;
    mlpack::data::StandardScaler scaler;

    arma::mat dataset;
    if (!mlpack::data::Load(dataset_filepath, dataset, true)) {
        throw std::runtime_error("Error loading dataset csv!");
    }
    arma::mat features = dataset.rows(0, dataset.n_rows - 2);
    arma::mat labels = dataset.row(dataset.n_rows - 1);

    std::cout << "Dataset " << dataset_filepath << " loaded. Contains "
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

//create optimizer from JSON config
ens::Adam create_optimizer(const nlohmann::json &optimizer_config) {
    return ens::Adam(
        optimizer_config["learning_rate"], optimizer_config["batch_size"],
        optimizer_config["momentum_decay"], optimizer_config["squared_gradient_decay"],
        optimizer_config["epsilon"], optimizer_config["max_iterations"],
        optimizer_config["tolerance"], true // shuffle data every epoch
    );
}

//mlp evaluation, prints r2 score
void evaluate_mlp(
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> &model,
    const arma::mat &X, const arma::mat &Y, const std::string &label) {
    arma::mat pred;
    model.Predict(X, pred);
    double r2 = R2Score(Y.row(0), pred.row(0));
    std::cout << label << " R² Score: " << r2 << std::endl;
}

//model evaluation (returns R²)
double return_r2(
    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> &model,
    const arma::mat &X, const arma::mat &Y, const std::string &label) {
    arma::mat pred;
    model.Predict(X, pred);
    double r2 = R2Score(Y.row(0), pred.row(0));
    std::cout << label << " R² Score: " << r2 << std::endl;
    return r2;
}

//save model and scaler
void save_model_and_scaler(
    const std::string &op_name,
    const mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> &model,
    const mlpack::data::StandardScaler &scaler) {
    std::string mlp_filename = op_name + "_model.bin";
    std::string scaler_filename = op_name + "_scaler.bin";
    mlpack::data::Save(mlp_filename, "model", model, true);
    mlpack::data::Save(scaler_filename, "scaler", scaler, true);
    std::cout << "Model saved as " << mlp_filename << std::endl;
    std::cout << "Scaler saved as " << scaler_filename << std::endl;
}
