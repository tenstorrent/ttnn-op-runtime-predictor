<div align="center">

<h1>TTNN-Op-Runtime-Predictor</h1>

</div>

## Overview

TTNN-Op-Runtime-Predictor contains models which predict the runtime of TTNN ops and has a C++ / Python API to access them. These models are small MLPs (Multi Layer Perceptrons) trained using datasets generated from the [tt-metal](https://github.com/tenstorrent/tt-metal) [sweep framework](https://github.com/tenstorrent/tt-metal/blob/main/tests/README.md). Each model is trained on a specific TTNN op using [mlpack](https://www.mlpack.org/doc/index.html), a C++ machine learning library. The op's input parameters are passed in and a runtime prediction is returned. These models can be queried for runtime predictions using an API in the repo that accepts JSON serialized op parameters.

## Using the Python API

To see examples using the python API to query models on the repo, see [interface-pybind/usage.py](interface-pybind/usage.py).

## Available Models

| TTNN Op                                  | Device        | TT-Metal Commit                                                                                                                      | R²        |
|------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------|-----------|
| ttnn.transformer.concatenate_heads       | Wormhole n150 | [20b59dcd0cbf63f2c5a9269fbbe217f715b4211d](https://github.com/tenstorrent/tt-metal/commit/20b59dcd0cbf63f2c5a9269fbbe217f715b4211d)  | 0.979006  |
| ttnn.experimental.create_qkv_heads       | Wormhole n150 | [20b59dcd0cbf63f2c5a9269fbbe217f715b4211d](https://github.com/tenstorrent/tt-metal/commit/20b59dcd0cbf63f2c5a9269fbbe217f715b4211d)  | 0.998757  |
| ttnn.exp                                 | Wormhole n150 | [87144607f757092c2c0cc817d12a8942d30fbfc9](https://github.com/tenstorrent/tt-metal/commit/87144607f757092c2c0cc817d12a8942d30fbfc9)  | 0.95      |

## Model Regeneration

TTNN-Op-Runtime-Predictor contains scripts for dataset generation and model training / retraining. 

1. **Generate Training Data:** Run sweep in tt-metal to collect op runtime. Models are trained on device kernel duration [ns] obtained from running sweeps with flag `--device-perf`.
2. **Preprocess Data:** Use script `create_dataset.py` in [train/python/model-regeneration](train/python/model-regeneration/) to aggregate collected data into a dataset.
3. **Train Models:** Use scripts `train_new_mlp.cpp` or `retrain_mlp.cpp`  in [train/mlpack/model-regeneration](train/mlpack/model-regeneration) to train or retrain MLPs using the processed datasets. 