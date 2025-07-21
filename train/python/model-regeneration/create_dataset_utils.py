import json

def parse_ints_from_string(text):
    text = text.strip("[]")
    numbers = text.split(",")

    result = []
    for number in numbers:
        number = number.strip()
        if number:
            result.append(int(number))
    return result

def load_json(sweep_results, sweep_test_vectors):

    with open(sweep_results) as results_file, open(sweep_test_vectors) as test_vectors_file:

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
        kernel_duration.append(int(device_perf_dict["DEVICE KERNEL DURATION [ns]"]))

    return kernel_duration