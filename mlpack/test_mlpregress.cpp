
#include <mlpack.hpp>

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ann.hpp>
#include <chrono>

using namespace mlpack;
//using namespace mlpack::ann;
using namespace std::chrono;

/*
// Function to calculate R² score
double R2Score(const arma::rowvec& actual, const arma::rowvec& predicted)
{
    double ssRes = arma::accu(arma::square(actual - predicted));
    double ssTot = arma::accu(arma::square(actual - arma::mean(actual)));
    return 1 - (ssRes / ssTot);
}
*/

int main(int argc, char** argv)
{
    bool train_only = false, predict_only = false;
    if (argc > 1) {
        if (strcmp(argv[1], "--train") == 0) {
            train_only = true;
        } else if (strcmp(argv[1], "--predict") == 0) {
            predict_only = true;
        } else {
            std::cerr << "Unrecognized argument, running both training and predicting." << std::endl;
        }
    }

    // Enable verbose logging
    Log::Info.ignoreInput = false;
    Log::Warn.ignoreInput = false;
    Log::Debug.ignoreInput = false;  // Only visible with -DDEBUG during compilation


    // Define MLP model with predefined structure
    FFN<MeanSquaredError, RandomInitialization> model;
    //FFN<> model;

    model.InputDimensions() = std::vector<size_t>({4}); // Input size = 4 features

    // Add layers: 4-input → 64 → 64 → 64 → 1-output
    model.Add<Linear>(128);
    model.Add<ReLU>();
    model.Add<Linear>(128);
    model.Add<ReLU>();
    model.Add<Linear>(128);
    model.Add<ReLU>();
    model.Add<Linear>(128);
    model.Add<ReLU>();
    model.Add<Linear>(1);

    // Initialize with random weights (placeholders)
    model.Reset();


    // Load dataset
    arma::mat dataset;
    if (!data::Load("matmul_height_sharded.csv", dataset, true))  // `true` for CSV header support
    {
        std::cerr << "Error loading data.csv!" << std::endl;
        return -1;
    }
    std::cout << "dataset.n_rows: " << dataset.n_rows << std::endl;
    std::cout << "dataset.n_cols: " << dataset.n_cols <<std::endl;

    // Extract features (first 4 cols) and labels (last col)
    arma::mat features = dataset.rows(0, 3);  // First 4 cols
    arma::mat labels = dataset.row(4);        // Last col (output)

    if (!predict_only) {
        // Set up optimizer (Adam with learning rate 0.01)
        ens::Adam optimizer(0.01,  // Learning rate
                            128,    // Batch size
                            0.9,   // Momentum decay
                            0.999, // Squared gradient decay
                            1e-8,  // Small epsilon to avoid division by zero
                            features.n_cols * 1000, // Max iterations (100 epochs)
                            1e-5,  // Tolerance (stopping criterion)
                            true); // Shuffle data every epoch

        // Train model
        auto start = high_resolution_clock::now();
        model.Train(features, labels, optimizer);
        auto end = high_resolution_clock::now();

        // Training time
        double totalTime = duration_cast<milliseconds>(end - start).count();
        std::cout << "Training completed in " << totalTime / 1000.0 << " seconds.\n";

        // Save trained model
        data::Save("mlp_model.bin", "model", model, true); // `true` for binary format
        std::cout << "Model saved as mlp_model.bin\n";
    }

    if (!train_only) {
        // load data from file
        if (predict_only) {
            if (!data::Load("mlp_model.bin", "model", model)) {
                throw std::runtime_error("Could not read model!");
            }
        }

        // Test input: Single sample with 4 features
        arma::mat input = arma::randu<arma::mat>(4, 1);
        arma::mat output;

        // Warm-up runs (avoid first-run overhead)
        for (int i = 0; i < 10; i++)
            model.Predict(input, output);

        // Benchmarking
        int iterations = 10000; // Run multiple predictions
        auto start = high_resolution_clock::now();

        for (int i = 0; i < iterations; i++)
            model.Predict(input, output);

        auto end = high_resolution_clock::now();

        // Compute time per prediction
        double totalTime = duration_cast<nanoseconds>(end - start).count() / 1e6; // Convert to milliseconds
        double avgTime = totalTime / iterations;

        // Output results
        std::cout << "Total Time: " << totalTime << " ms for " << iterations << " predictions.\n";
        std::cout << "Average Time per Prediction: " << avgTime << " ms (" << avgTime * 1000 << " µs)\n";

        // **Make Predictions for Training Data**
        //arma::mat predictions;
        //model.Predict(features, predictions);
        //Log::Info << "Predictions completed." << std::endl;

        // **Calculate and Display R² Score**
        //double r2Score = R2Score(labels, predictions.row(0));
        mlpack::R2Score<false> r2;
        arma::rowvec l = labels.row(0);
        double r2Score = r2.Evaluate(model, features, l);
        Log::Info << "R² Score on Training Data: " << r2Score << std::endl;
    }

    return 0;
}
