import numpy as np
from automaton_handler import Automaton
import os
import tensorflow as tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Automaton2Network:
    def __init__(self, automaton, train_set_size):
        """
        Creates a base for the creation of a network
        :param automaton: automaton to be converted
        :param train_set_size: size of the set of words used for training the model (int)
        """
        self.automaton = automaton
        self.x_train, self.y_train = self.automaton.classify_words_derivation(
            train_set_size
        )

    def create_convolut_model(
        self,
        nb_layers=32,
        nb_kernels=4,
        pooling_kernel=4,
        epochs_amount=15,
        save=True,
    ):
        """
        This function creates a sequential model reproducing the behavior of the automaton
        :param nb_layers: amount of different nodes used for convolution (int)
        :param nb_kernels: number of entry accessible by each node in the convolution layer (int)
        :param pooling_kernel: number of node affected to one pooling node (int)
        :param epochs_amount: number of iteration of the training sequence (int)
        :param save: choose if the model is saved or not after generation (bool)
        :param verbose: choose if you want to print execution information (bool)
        :return: list of epoch validation accuracy
        """
        print(
            "Creating model for the regular expression: {}".format(
                self.automaton.get_regex()
            )
        )
        inp = np.shape(self.x_train[0])
        model = tf.keras.Sequential()
        model.add(
            tf.keras.layers.Conv1D(
                nb_layers, nb_kernels, input_shape=inp, padding="same"
            )
        )
        model.add(tf.keras.layers.Activation(tf.keras.activations.relu))
        model.add(
            tf.keras.layers.MaxPooling1D(
                pool_size=pooling_kernel, padding="same"
            )
        )
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(units=1))
        model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))
        print(model.summary())
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs_amount,
            batch_size=32,
            validation_split=0.2,
        )
        accuracies = history.history["accuracy"]
        if save:
            if not os.path.exists("./stored_models"):
                os.makedirs("./stored_models")
            num = len([name for name in os.listdir("./stored_models")]) + 1
            model.save("./stored_models/saved_model_{}".format(num))
            print("Model saved at 'stored_models/saved_model_{}'".format(num))
        return accuracies

    def create_recurrent_model(
        self,
        nb_nodes=64,
        epochs_amount=15,
        save=True,
    ):
        # print(
        #     "Creating model for the regular expression: {}".format(
        #         self.automaton.get_regex()
        #     )
        # )
        inp = np.shape(self.x_train[0])
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.GRU(nb_nodes))
        model.add(tf.keras.layers.Dense(units=1))
        model.add(tf.keras.layers.Activation(tf.keras.activations.sigmoid))
        # print(model.summary())
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        history = model.fit(
            self.x_train,
            self.y_train,
            epochs=epochs_amount,
            batch_size=32,
            validation_split=0.2,
        )
        accuracies = history.history["accuracy"]
        if save:
            if not os.path.exists("./stored_models"):
                os.makedirs("./stored_models")
            num = len([name for name in os.listdir("./stored_models")]) + 1
            model.save("./stored_models/saved_model_{}".format(num))
            print("Model saved at 'stored_models/saved_model_{}'".format(num))
        return accuracies

    @staticmethod
    def load_model(number):
        """
        This function loads a saved model
        :param number: number of the saved model
        :return: usable model
        """
        return tf.keras.models.load_model(
            "./stored_models/saved_model_{}".format(number)
        )
    

    @staticmethod
    def get_accuracy(automata , dataset = 5000):
        net = Automaton2Network(automata, dataset)
        acc = net.create_recurrent_model(epochs_amount = 1, save=False)
        return acc[-1]



if __name__ == "__main__":
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        print("running on, ", tf.test.gpu_device_name())
    else:
        print("No GPU found. Running on CPU")
    auto = Automaton.random_automaton({"a", "b"}, 5, 10, 0.12, 2, 2, 1, 1)
    net = Automaton2Network(auto, 20000)
    c = net.create_convolut_model(epochs_amount=4, save=True)
    r = net.create_recurrent_model(epochs_amount=4)
    print(c, r)

