# SPDX-FileCopyrightText: (c) 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import numpy as np


from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from multiprocessing import Pool

import csv

from tqdm import tqdm

def h_net_gen(num_layers, layer_sizes):
    if (num_layers == 0):
        yield []
    else:
        for s in layer_sizes:
            layers = [2**s]
            for hl in h_net_gen(num_layers-1, layer_sizes):
                new_layers = layers + hl
                yield new_layers

def get_network_score(hidden_layers, X_trainscaled, X_testscaled, y_train, y_test):
    reg = MLPRegressor(hidden_layer_sizes=hidden_layers,activation="relu" ,random_state=1, max_iter=5000).fit(X_trainscaled, y_train)
    y_pred=reg.predict(X_testscaled)
    score = r2_score(y_pred, y_test)
    return score

def get_network_score_packed(args):
    hidden_layers, X_trainscaled, X_testscaled, y_train, y_test = args
    return get_network_score(hidden_layers,X_trainscaled, X_testscaled, y_train, y_test)

if __name__ == '__main__':
    f = open("matmul_height_sharded.csv")
    f.readline()  # skip the header
    data = np.loadtxt(f, delimiter=",")
    #print(data)

    X = data[:, 0:4]
    y = data[:, 4]

    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1, test_size=0.2)

    sc_X = StandardScaler()
    X_trainscaled=sc_X.fit_transform(X_train)
    X_testscaled=sc_X.transform(X_test)


    #for nets in h_net_gen(3, [3, 4, 5, 6, 7]):
    #    print(f"net: {nets}")
    test_networks = []
    for l in [2,3,4,5]:
    #for l in [2,3]:
        for hidden_layers in h_net_gen(l, [3, 4, 5, 6, 7]):
            #score = get_network_score(hidden_layers)
            #print(f"Layers: {l} r2 score: {score}, hidden_layers={hidden_layers}")
            #networks.append((hidden_layers, score))
            test_networks.append(hidden_layers)
            
    with Pool() as p:
        #result = list(tqdm(p.starmap(get_network_score, [(n, X_trainscaled, X_testscaled, y_train, y_test) for n in test_networks])))
        tasks = [(n, X_trainscaled, X_testscaled, y_train, y_test) for n in test_networks]
        result = list(tqdm(p.imap_unordered(get_network_score_packed, tasks), total=len(tasks)))

    networks = list(zip(test_networks, result))
    networks.sort(key=lambda tup: tup[1], reverse=True)
    for n, s in networks[:10]:
        print(f"network: {n}, score: {s}")

    with open("mlp_network_results_2.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(networks)