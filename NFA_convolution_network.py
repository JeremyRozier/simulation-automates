import numpy as np
from automaton_handler import Automaton
import tensorflow as ts


class Automaton2Network:
    def __init__(self, automaton, train_set_size):
        """
        Creates a base for the creation of a network
        :param automaton:
        :param trainset_size:
        """
        self.automaton = automaton
        self.train_set = self.automaton.classify_words(train_set_size)
        self.network = None
        self.x_train, self.y_train = np.hsplit(self.train_set, 2)

    def create_network(self, nb_filters=32, nb_kernels=4, pooling_kernel=4):
        model = ts.keras.Sequential()
        model.add(ts.keras.layers.Conv1D(nb_filters,
                                         nb_kernels,
                                         input_shape=(len(self.train_set[-1][0]),
                                                      len(self.automaton.input_symbols))
                                         )
                  )
        model.add(ts.keras.layers.Activation(ts.keras.activations.relu))
        model.add(ts.keras.layers.MaxPooling1D(pool_size=pooling_kernel))
        model.add(ts.keras.layers.Flatten())
        model.add(ts.keras.layers.Dense(units=2))
        model.add(ts.keras.layers.Activation(ts.keras.activations.softmax))
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])
        model.fit(self.x_train, self.y_train, epochs=50, batch_size=32, validation_split=0.2)
        self.network = model
