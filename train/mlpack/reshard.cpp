// SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <armadillo>

using namespace std;

int main()
{

    std::string subset = "WIDTH_ROW_MAJOR_HEIGHT_ROW_MAJOR";
    arma::mat data;
    mlpack::data::Load("reshard_data/" + subset + ".csv", data, true);
    
    arma::mat trainData;
    arma::mat validationData;
    arma::mat testData;
    arma::mat temp;
    
    // Enable verbose logging
    mlpack::Log::Info.ignoreInput = false;
    mlpack::Log::Warn.ignoreInput = false;
    mlpack::Log::Debug.ignoreInput = false;  // Only visible with -DDEBUG during compilation

    mlpack::data::Split(data, trainData, temp, 0.3);
    mlpack::data::Split(temp, validationData, testData, 0.5);

    arma::mat features = trainData.rows(0, 4);
    arma::mat labels = trainData.row(5);        // Last col (output)
    arma::mat validFeatures = validationData.rows(0, 4);
    arma::rowvec validLabels = validationData.row(5);

    // Apply StandardScaler
    mlpack::data::StandardScaler scaler;
    scaler.Fit(features);
    scaler.Transform(features, features);

    std::cout<<"train: "<<to_string(trainData.n_cols) << " x " << to_string(trainData.n_rows)<<"\n";
    std::cout<<"validation: "<<to_string(validationData.n_cols) << " x " << to_string(validationData.n_rows)<<"\n";
    std::cout<<"test: "<<to_string(testData.n_cols) << " x " << to_string(testData.n_rows)<<"\n";

    mlpack::FFN<mlpack::MeanSquaredError, mlpack::RandomInitialization> model;
    model.InputDimensions() = std::vector<size_t>({5});

    model.Add<mlpack::Linear>(100);
    model.Add<mlpack::ReLU>();
    model.Add<mlpack::Linear>(50);
    model.Add<mlpack::ReLU>();
    model.Add<mlpack::Linear>(1);

    model.Reset();

    ens::Adam optimizer(0.001, 256, 0.9, 0.999, 1e-8, features.n_cols * 200, 1e-5);
    model.Train(features, labels, optimizer,
                ens::ProgressBar(), ens::PrintLoss());
    mlpack::R2Score<false> r2;
    arma::rowvec l = labels.row(0);
    double r2Score = r2.Evaluate(model, features, l);
    mlpack::Log::Info << "R² Score on Training Data: " << r2Score << std::endl;
    
    scaler.Transform(validFeatures, validFeatures);
    r2Score = r2.Evaluate(model, validFeatures, validLabels);
    mlpack::Log::Info << "R² Score on Validation Data: " << r2Score << std::endl;

    // Save the model and scaler
    mlpack::data::Save("model_" + subset + ".bin", "model", model, true);
    mlpack::data::Save("scaler_" + subset + ".bin", "scaler", scaler, true);
    
    return 0;
}
