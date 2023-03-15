import random
import numpy as np
from automaton_handler import Automaton
from NFA_convolution_network import *
import os
import tensorflow as tf
import matplotlib.pyplot as plt
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def test_parameter(
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
        acc = net.create_convolut_model(epochs_amount=4)

        accuracies.append(acc[-1])

    # plot accuracy curve
    plt.plot(param_values, accuracies)
    plt.xlabel("amount of" + param_name)
    plt.ylabel("Accuracy at the end of training")
    plt.show()

test_parameter("trans_density", [0.12, 0.3, 0.5], {"a", "b"}, 5, 10, 0.12, 2, 2, 1, 1)
# test_parameter("alphabet", [2, 3, 4], {"a", "b"}, 5, 10, 0.12, 2, 2, 1, 1)
