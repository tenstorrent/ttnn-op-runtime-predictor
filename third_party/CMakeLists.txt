include(ExternalProject)

ExternalProject_Add(
    mlpack
    CMAKE_GENERATOR "Unix Makefiles"
    CMAKE_ARGS
    -DBUILD_CLI_EXECUTABLES=ON
    -DCMAKE_INSTALL_PREFIX=${CMAKE_BINARY_DIR}
    GIT_REPOSITORY https://github.com/mlpack/mlpack.git
    GIT_TAG "4.6.2"
    GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
)
