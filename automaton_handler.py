import random
from automata.fa.nfa import NFA
from automata.fa.gnfa import GNFA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import exrex
import numpy as np
import pickle


class Automaton(NFA):
    def __init__(
            self,
            states,
            input_symbols,
            transitions,
            initial_state,
            final_states
    ):
        super().__init__(
            states=states,
            input_symbols=input_symbols,
            transitions=transitions,
            initial_state=initial_state,
            final_states=final_states
        )

    def __setattr__(self, name, value):
        """Overrides __setattr__ to allow making other attributes"""
        pass

    @staticmethod
    def random_automaton(
            alphabet,
            min_states,
            max_states,
            trans_density,
            init_amount,
            final_amount,
            init_range=0,
            final_range=0,
            seed=None
    ):
        """
        This function creates a random
        automaton with desired probabilities
        :param alphabet: set of the alphabet used by the automaton (set of str)
        :param min_states: minimum amount of states (int)
        :param max_states: maximum amount of states (int)
        :param trans_density: chance for a transition
        to be created (percentage)
        :param init_amount: amount of initial states (int)
        :param final_amount: amount of final states (int)
        :param init_range: maximal variation
        in the amount of initial states (int)
        :param final_range: maximal variation in
        the amount of final states (int)
        :param seed: random seed, None by default
        :return: a set of states, a dictionary of transition,
        set of initial states, set of final states
        """
        if seed is not None:
            random.seed(seed)
        states = {"init"}
        for i in range(random.randint(min_states, max_states)):
            states.add(i)
        trans = {"init": {"": set()}}
        for state_in in states.difference({"init"}):
            for state_out in states.difference({"init"}):
                for char in alphabet:
                    if random.random() < trans_density:
                        try:
                            trans[state_in][char].add(state_out)
                        except KeyError:
                            try:
                                trans[state_in][char] = {state_out}
                            except KeyError:
                                trans[state_in] = {char: {state_out}}
        init = set()
        for i in range(
                max(
                    init_amount + random.randint(-init_range, init_range),
                    1,
                )
        ):
            if len(states.difference(init.union({"init"}))) > 0:
                init.add(
                    random.choice(
                        list(states.difference(init.union({"init"})))
                    )
                )
            else:
                break
        for i in init:
            trans["init"][""].add(i)
        final = set()
        for i in range(
                max(
                    final_amount + random.randint(-final_range, final_range),
                    0,
                )
        ):
            if len(states.difference(final.union({"init"}))) > 0:
                final.add(
                    random.choice(
                        list(states.difference(final.union({"init"})))
                    )
                )
            else:
                break
        return Automaton(states, alphabet, trans, "init", final)

    def __str__(self):
        try:
            open("./display.png", "x")
        except FileExistsError:
            pass
        self.show_diagram(path="./display.png")
        plt.figure("Automate")
        plt.axis("off")
        plt.imshow(mpimg.imread("display.png"))
        plt.show()
        return super.__str__(self)

    def get_one_hot_index(self):
        return {list(self.input_symbols)[i]: i for i in range(len(self.input_symbols))}

    def one_hot_encoder(self, word, length):
        """
        This function encode a word in the one-hot format
        :param word: word to encode (str)
        :param length: length of the one-hot array used for encoding
        :return: array with shape (len(self.input_symbols), length) filled with 0 and 1
        """
        encoded = np.zeros((len(self.input_symbols), length))
        for i in range(len(word)):
            encoded[self.get_one_hot_index()[word[i]]][i] = 1
        return encoded

    def one_hot_decoder(self, encoded):
        """
        This function returns the word encoded in a one-hot array
        :param encoded: one-hot array to decode
        :return: decoded word (str)
        """
        word = ""
        one_hot_index = tuple(self.get_one_hot_index().keys())
        for col in range(np.shape(encoded)[1]):
            index = np.argwhere(np.hsplit(encoded, np.shape(encoded)[1])[col].reshape((np.shape(encoded)[0],)))
            if np.shape(index)[0]:
                word += one_hot_index[index[0][0]]
            else:
                break
        return word

    def classify_words(self, nb):
        """
        This function creates a 2 array. One with shape (nb, alphabet_length, word_max_length) where nb in the number of
        words classified, alphabet_length is the amount of character in the alphabet and word_max_length is the length
        of the longest word classified. Ths array stores all the words classified encoded in one hot format.
        The second array with shape (nb, ) stores a one in it's x index if the x classified word if accepted by the
        automaton and a 0 otherwise.
        :param nb: number of word to classify
        :return: 2 arrays with shape (nb, alphabet_length, word_max_length) and (nb, ) and dtype=float64
        """
        classified = 0
        sigma_star = Automaton(
            {0},
            self.input_symbols,
            {0: {char: {0} for char in self.input_symbols}},
            0,
            {0}
        ).get_regex()
        one_hot_words = []
        tag = []
        length = round(np.log(nb) / np.log(len(self.input_symbols))) + 1
        for word in exrex.generate(sigma_star, limit=100):
            encoded = self.one_hot_encoder(word, length)
            one_hot_words.append(encoded)
            tag.append(int(self.accepts_input(word)))
            classified += 1
            if classified >= nb:
                return np.array(one_hot_words, dtype="float64"), np.array(tag, dtype="float64")

    def get_regex(self):
        """
        :return: regular expression of the automaton (str)
        """
        return GNFA.from_nfa(self).to_regex()

    def save_automaton(self, name):
        """
        This function saves the automaton in a pickle file
        :param name: name of the automaton (str)
        :return: None
        """
        file_name = "{}.pkl".format(name)
        with open(file_name, "wb") as file:
            pickle.dump((self.states,
                         self.input_symbols,
                         self.transitions,
                         self.initial_state,
                         self.final_states)
                        , file)
            print("Automaton saved in {}".format(file_name))

    @staticmethod
    def load_automaton(file_name):
        """
        This function load a saved automaton
        :param file_name: name of the file containing the automaton (str)
        :return: Automaton object
        """
        with open(file_name, "rb") as file:
            info = pickle.load(file)
            return Automaton(info[0], info[1], info[2], info[3], info[4])


if __name__ == "__main__":
    alphabet = {"a", "b"}
    min_states = 5
    max_states = 10
    trans_density = 0.12
    init_amount = 2
    final_amount = 2
    init_range = 1
    final_range = 1
    automaton = Automaton.random_automaton(
        alphabet,
        min_states,
        max_states,
        trans_density,
        init_amount,
        final_amount,
        init_range,
        final_range,
    )
