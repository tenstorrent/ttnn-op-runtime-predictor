# TTNN Op Runtime Predictor

<div align="center">

<h1>

[Hardware](https://tenstorrent.com/cards/) | [Discord](https://discord.gg/tenstorrent) | [Join Us](https://boards.greenhouse.io/tenstorrent?gh_src=22e462047us) 

</h1>

<br>

**ttnn-op-runtime-predictor** is a framework for training and querying empirical performance models for [TTNN](https://docs.tenstorrent.com/tt-metal/latest/ttnn/index.html) tensor operations. 


</div>

----
# What is this repo?
TTNN Op Runtime Predictor is a framework for training and querying empirical performance models for TTNN tensor operations. 

This is accomplished by generating a dataset for each operation using the [tt-metal sweep framework](https://github.com/tenstorrent/tt-metal/blob/main/tests/README.md) to collect runtimes for various parameterizations. Then, the scripts in this repo postprocess the collected data and and train small MLPs (Multi Layer Perceptrons) using [mlpack](https://www.mlpack.org/doc/index.html). Each TTNN operation must be modeled separately. The trained models are committed to this repo, and a list of currently available models is below.

This repo also contains a C++ and Python API for querying the trained models. The op's input parameters are passed in as JSON and a runtime prediction is returned.


-----
# Related Tenstorrent Projects
- [tt-mlir](https://github.com/tenstorrent/tt-mlir)
- [tt-metalium](https://github.com/tenstorrent/tt-metal)

-----
# Getting Started
**Environment setup and build instructions to be added here.**

-----
# Using the Python API (Experimental)
**Python API build and installation instructions to be added here**

To see examples using the python API to query models on the repo, see [interface-pybind/usage.py](interface-pybind/usage.py).

# Available Operation Models

| TTNN Op                                  | Device        | TT-Metal Commit                                                                                                                      | R²        |
|------------------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------------------|-----------|
| ttnn.transformer.concatenate_heads       | Wormhole n150 | [20b59dcd0cbf63f2c5a9269fbbe217f715b4211d](https://github.com/tenstorrent/tt-metal/commit/20b59dcd0cbf63f2c5a9269fbbe217f715b4211d)  | 0.979006  |
| ttnn.experimental.create_qkv_heads       | Wormhole n150 | [20b59dcd0cbf63f2c5a9269fbbe217f715b4211d](https://github.com/tenstorrent/tt-metal/commit/20b59dcd0cbf63f2c5a9269fbbe217f715b4211d)  | 0.998757  |
| ttnn.exp                                 | Wormhole n150 | [87144607f757092c2c0cc817d12a8942d30fbfc9](https://github.com/tenstorrent/tt-metal/commit/87144607f757092c2c0cc817d12a8942d30fbfc9)  | 0.95      |

# Model Regeneration

The operation models in this repo are only valid as of the commit they are trained on. Changes to the operation or the runtime infrastructure could change the performance characteristics of an operation. The framework in TTNN Op Runtime Predictor is built with this in mind to allow few-button (i.e. mostly automated) model regeneration. 

1. **Generate Training Data:** Run sweeps in tt-metal to collect op runtime. Models are trained on device kernel duration (in nanoseconds) obtained from running sweeps with flag `--device-perf`.
2. **Preprocess Data:** Use script `create_dataset.py` in [train/python/model-regeneration](train/python/model-regeneration/) to aggregate collected data into a dataset.
3. **Train Models:** Use scripts `train_new_mlp.cpp` or `retrain_mlp.cpp`  in [train/mlpack/model-regeneration](train/mlpack/model-regeneration) to train or retrain MLPs using the processed datasets. 