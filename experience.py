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

def training_test(
    max_training_amount,
    discretion_amount,
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
    data_epoch = max_training_amount//discretion_amount
    accuracies = []
    data_set = []
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
    data_count = data_epoch
    automaton = Automaton.random_automaton(**params)
    for i in range(discretion_amount):
        data_set.append(data_count)
        net = Automaton2Network(automaton, data_set[-1])
        acc = net.create_recurrent_model(epochs_amount = 4, save=False)
        accuracies.append(acc[-1])
        data_count = data_epoch + data_set[-1]


    # plot accuracy curve
    plt.plot(data_set, accuracies)
    plt.xlabel("amount of epoch" )
    plt.ylabel("Accuracy at the end of training")
    plt.show()

# parameter_test("trans_density", [0.2 for i in range(100)], {"a", "b"}, 5, 10, 0.12, 2, 2, 1, 1)
# parameter_test("max_states", [6,7,8,9,10,11,12,13,14,15], {"a", "b"}, 5, 10, 0.12, 2, 2, 1, 1)
training_test(500,25, {"a", "b"}, 5, 15, 0.12, 2, 2, 1, 1)



