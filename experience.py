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

        net = Automaton2Network(automaton, 200)
        acc = net.create_model(epochs_amount=1, save=False, verbose=False)

        accuracies.append(acc[-1])
    accuracy = np.average((np.array(accuracies)))
    print(accuracy)


    """# plot accuracy curve
    plt.plot(param_values, accuracies)
    plt.xlabel("amount of" + param_name)
    plt.ylabel("Accuracy at the end of training")
    plt.show()"""

parameter_test("trans_density", [0.2 for i in range(100)], {"a", "b"}, 5, 10, 0.12, 2, 2, 1, 1)
# test_parameter("alphabet", [2, 3, 4], {"a", "b"}, 5, 10, 0.12, 2, 2, 1, 1)


"""X = [0, 0.1, 0.15, 0.175, 0.1875, 0.2, 0.225, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
Y = [1.0, 0.9551799994707107, 0.9231600013375282, 0.8959400001168251, 0.898860000371933, 0.895760001540184,
     0.933140001296997, 0.9196100005507469, 0.9536899998784065, 0.971180000603199, 0.995469999909401, 1.0, 1.0, 1.0,
     1.0, 1.0]

plt.plot(X, Y)
plt.show()
"""
