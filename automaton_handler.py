import random
from automata.fa.nfa import NFA
from automata.fa.gnfa import GNFA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import exrex
import numpy as np


class Automaton(NFA):
    def __init__(
        self,
        states,
        input_symbols,
        transitions,
        initial_state,
        final_states,
    ):
        super().__init__(
            states=states,
            input_symbols=input_symbols,
            transitions=transitions,
            initial_state=initial_state,
            final_states=final_states,
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
        seed=None,
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

    def classify_words(self, nb):
        """
        This function creates a 2 dimensional array with shape (np, 2) where
        the first column is a word in the alphabet
        and the second column is "1" if the word on the same row is accepted
        by the automaton and "0" otherwise
        :param nb: number of word in the array
        :return: 2 dimensional array with shape (np, 2) and datatype=str
        """
        classified = 0
        sigma_star = Automaton(
            {0},
            self.input_symbols,
            {0: {char: {0} for char in self.input_symbols}},
            0,
            {0},
        ).get_regex()
        classification = []
        for word in exrex.generate(sigma_star, limit=100):
            classification.append([word, int(self.accepts_input(word))])
            classified += 1
            if classified >= nb:
                return np.array(classification)

    def get_regex(self):
        return GNFA.from_nfa(self).to_regex()


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
    print(
        [
            automaton.accepts_input(word)
            for word in automaton.get_random_declined_words(10)
        ]
    )
    print(automaton)
