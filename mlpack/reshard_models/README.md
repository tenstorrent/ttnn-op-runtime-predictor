# Models

These models predict the kernel duration for the reshard operation.

Model inputs:
- input grid size x and y
- output grid size x and y
- size of the matrix, in number of BF16 tiles

Model output:
Kernel duration, in nanoseconds

Naming scheme: `WIDTH_ROW_MAJOR_HEIGHT_ROW_MAJOR` means the model for reshards from `width` sharded, in `row major` orientation, to `height` sharded, in `row major` orientation.

# Model Architecture
Each model is composed of two components:
1. A standard scalar (to ensure 0 mean and unit variance for inputs to the MLP)
2. A small MLP

# Model Implementation
Production models are implemented and trained using `mlpack` 4.5.1, a header-only machine learning library in C++.

| Model                               | MLP Architecture                                                                         | Training Details                             | R^2    |
|-------------------------------------|------------------------------------------------------------------------------------------|----------------------------------------------|--------|
| Block Row Major -> Block Row Major  | `mlpack::Linear(100), mlpack::ReLU, mlpack::Linear(50), mlpack::ReLU, mlpack::Linear(1)` | Adam, 200 epochs, batch size 256, 0.001 lr   | 0.972  |
| Width Row Major -> Block Row Major  | `mlpack::Linear(100), mlpack::ReLU, mlpack::Linear(50), mlpack::ReLU, mlpack::Linear(1)` | Adam, 200 epochs, batch size 256, 0.001 lr   | 0.951  |
| Height Row Major -> Block Row Major | `mlpack::Linear(100), mlpack::ReLU, mlpack::Linear(50), mlpack::ReLU, mlpack::Linear(1)` | Adam, 200 epochs, batch size 256, 0.001 lr   | 0.977  |
| Height Row Major -> Width Row Major | `mlpack::Linear(100), mlpack::ReLU, mlpack::Linear(50), mlpack::ReLU, mlpack::Linear(1)` | Adam, 200 epochs, batch size 256, 0.001 lr   | 0.975  |
| Width Row Major -> Height Row Major | `mlpack::Linear(100), mlpack::ReLU, mlpack::Linear(50), mlpack::ReLU, mlpack::Linear(1)` | Adam, 200 epochs, batch size 256, 0.001 lr   | 0.891  |
| Block Row Major -> Height Row Major | See notes                                                                                | Adam, 1000 epochs, batch size 1024, 0.0001 lr| 0.862* |
| Block Row Major -> Width Row major  | See notes                                                                                |                                              |        |

Note:
- Block Row Major -> Height Row Major and Block Row Major -> Width Row Major currently does not converge in mlpack.
- Block Row Major -> Height Row major does converge when using scikit learn with `MLPRegressor(hidden_layer_sizes=(128, 128, 128), max_iter=1000, solver="adam", batch_size=1024, random_state=seed, verbose=1, learning_rate_init=0.0001)`
  - The same training conditions do not converge in mlpack, despite being robust to seed noise in scikit learn
  - I (@arminaleTT) am trying to implement methods to export weights from scikit learn and import into mlpack. This is non-trivial as there is no common format between the two. We need to export the weights to raw csvs and then manually lay vectorize them in the mlpack model
- I (@arminaleTT) have not found a model + training condition that converges for Block Row Major -> Width Row Major

The data files are too big to push to github. Contact @arminaleTT