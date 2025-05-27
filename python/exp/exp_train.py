import csv
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
import numpy as np

import joblib

def read_dataset(file_path, filter_condition):
    inputs = []
    outputs = []
    
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader) # skip header
        for row in reader:
            if row:  # Apply filter condition
                #inputs.append([int(x) for x in row[4:-3]] + [(int(row[-2]) * int(row[-3])) // (32 * 32)])  # Convert remaining inputs to float
                #outputs.append(float(row[-1]))  # Convert output to float
                if row[2] == "0":
                    inputs.append([(int(row[0]) // 32)] + [(int(row[1]) // 32)] + [int(x) for x in row[2:8]])
                else:
                    inputs.append([row[0]] + [(int(row[1]) // 32)] + [(int(row[2]) // 32)] + [int(x) for x in row[3:8]])
                outputs.append(float(row[-1]))
    
    return inputs, outputs

def train_mlp_regressor(inputs, outputs):
    seed = 20
    X_train, X_temp, y_train, y_temp = train_test_split(inputs, outputs, test_size=0.3, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed)
    
    scaler = StandardScaler()
    model = MLPRegressor(hidden_layer_sizes=(128, 128, 128), max_iter=1500, solver="adam", batch_size=512, random_state=seed, verbose=1, learning_rate_init=0.001)
    pipeline = Pipeline([
        ('scaler', scaler),
        ('mlp', model)
    ])
    pipeline.fit(X_train, y_train)
    
    train_score = pipeline.score(X_train, y_train)
    val_score = pipeline.score(X_val, y_val)
    # test_score = model.score(X_test, y_test)
    
    print(f"MLP Regressor R^2 Score - Train: {train_score}, Validation: {val_score}")
    
    return pipeline

def pkl_to_csv(path):
    pipeline = joblib.load(path)
    scaler_params = pipeline["scaler"]
    mean = scaler_params.mean_
    scale = scaler_params.scale_

    # Extract MLP parameters
    mlp_params = pipeline["mlp"]
    coefs = mlp_params.coefs_  # List of weight matrices
    intercepts = mlp_params.intercepts_  # List of bias vectors

    # Save scaler parameters as CSV
    np.savetxt("scaler_mean.csv", mean, delimiter=",")
    np.savetxt("scaler_scale.csv", scale, delimiter=",")

    # Save MLP weights and biases as CSV
    for i, coef in enumerate(coefs):
        np.savetxt(f"coefs_{i}.csv", coef, delimiter=",")
    for i, intercept in enumerate(intercepts):
        np.savetxt(f"intercepts_{i}.csv", intercept, delimiter=",")


if __name__ == "__main__":
    file_path = "big_exp_dataset.csv"  # Change this to your actual file path
    filter_condition = []  # Change filter condition as needed
    inputs, outputs = read_dataset(file_path, filter_condition)
    
    print("Inputs:")
    for inp in inputs[:5]:  # Show first 5 rows as example
        print(inp)
    for inp in inputs[8150:8155]:  # Show first 5 rows as example
        print(inp)
    print(f"num inputs: {len(inputs)}")
    
    print("\nOutputs:")
    print(outputs[:5])
    
    model = train_mlp_regressor(inputs, outputs)
    joblib.dump(model, 'exp_model.pkl')
    #pkl_to_csv("exp_model.pkl")