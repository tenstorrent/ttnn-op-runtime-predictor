#!/bin/bash

# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

g++ -O3 -std=c++20 -DMLPACK_ENABLE_ANN_SERIALIZATION=ON -DDEBUG -o test_mlpregress test_mlpregress.cpp -L/opt/homebrew/lib -larmadillo -I/opt/homebrew/include -L/opt/homebrew/lib
