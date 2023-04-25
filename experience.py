import random

import numpy as np

from NFA_convolution_network import *
import os
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def parameter_test(
    param_name,
    param_values,
    alphabet,
    min_states,
    max_states,
    trans_density,
    init_amount,
    final_amount,
    init_range=0,
    final_range=0,
    seed=None,
):
    accuracies = []
    for param_value in param_values:
        # create automaton with current param value
        params = {
            "alphabet": alphabet,
            "min_states": min_states,
            "max_states": max_states,
            "trans_density": trans_density,
            "init_amount": init_amount,
            "final_amount": final_amount,
            "init_range": init_range,
            "final_range": final_range,
            "seed": seed,
        }
        if param_name == "alphabet":
            alphabet = set()
            for i in range(param_value):
                letters = "abcdefghijklmnopqrstuvwxyz"
                random_letter = random.choice(letters)
                alphabet.add(random_letter)
            param_value = alphabet
        params[param_name] = param_value
        automaton = Automaton.random_automaton(**params)

        net = Automaton2Network(automaton, 20000)
        acc = net.create_convolut_model(epochs_amount=3, save=False)

        accuracies.append(acc[-1])
    accuracy = np.average((np.array(accuracies)))
    print(accuracy)


    # plot accuracy curve
    plt.plot(param_values, accuracies)
    plt.xlabel("amount of" + param_name)
    plt.ylabel("Accuracy at the end of training")
    plt.show()

def vector_test(
    min_size,
    max_size
):
    
    data_set = []
    for i in range(min_size, max_size):
        data_set.append(i)
    for s in range(1,3):
        accuracies = []
        automaton = Automaton.load_automaton(f'simulation-automates/stored_automatons/size_{s}/aut_1.pkl')
        for i in data_set:
            acc = Automaton2Network.get_accuracy(automaton, units = i)
            accuracies.append(acc)
        plt.plot(data_set, accuracies, label = f'{s} nodes' )


    # plot accuracy curve
    
    plt.xlabel("size of state vector" )
    plt.ylabel("Accuracy at the end of training")
    plt.show()

# parameter_test("trans_density", [0.2 for i in range(100)], {"a", "b"}, 5, 10, 0.12, 2, 2, 1, 1)
# parameter_test("max_states", [6,7,8,9,10,11,12,13,14,15], {"a", "b"}, 5, 10, 0.12, 2, 2, 1, 1)
vector_test(20,23)



