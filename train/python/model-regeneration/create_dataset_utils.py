# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import json

MAX_TENSOR_RANK = 4
NUM_SUPPORTED_DTYPES = 5
NUM_BUFFER_TYPES = 2

def parse_ints_from_string(text):
    text = text.strip("[]")
    numbers = text.split(",")

    numbers = [n.strip() for n in numbers]
    return [int(n) for n in numbers if n]

def load_json(sweep_test_vectors, sweep_results):

    with open(sweep_results, 'r') as results_file, open(sweep_test_vectors, 'r') as test_vectors_file:

        results = json.load(results_file)
        test_vectors = json.load(test_vectors_file)

    return results, test_vectors

#get kernel duration (runtime) from sweep results vectors
def get_kernel_durations(sweeps_results):

    kernel_duration = []
    for i in range(len(sweeps_results)):
        dict = sweeps_results[i]
        if "device_perf" not in dict.keys():
            kernel_duration.append(-1)
            continue
        device_perf_dict = dict["device_perf"]
        kernel_duration.append(float(device_perf_dict["DEVICE KERNEL DURATION [ns]"]))

    return kernel_duration
