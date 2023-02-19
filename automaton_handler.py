import random
from automata.fa.nfa import NFA
from automata.fa.gnfa import GNFA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import exrex
import numpy as np


class Automaton(NFA):
    def __init__(self, states, input_symbols, transitions, initial_state, final_states):
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
        This function creates a random automaton with desired probabilities
        :param alphabet: set of the alphabet used by the automaton (set of str)
        :param min_states: minimum amount of states (int)
        :param max_states: maximum amount of states (int)
        :param trans_density: chance for a transition to be created (percentage)
        :param init_amount: amount of initial states (int)
        :param final_amount: amount of final states (int)
        :param init_range: maximal variation in the amount of initial states (int)
        :param final_range: maximal variation in the amount of final states (int)
        :param seed: random seed, None by default
        :return: a set of states, a dictionary of transition, set of initial states, set of final states
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
        for i in range(max(init_amount + random.randint(-init_range, init_range), 1)):
            if len(states.difference(init.union({"init"}))) > 0:
                init.add(random.choice(list(states.difference(init.union({"init"})))))
            else:
                break
        for i in init:
            trans["init"][""].add(i)
        final = set()
        for i in range(max(final_amount + random.randint(-final_range, final_range), 0)):
            if len(states.difference(final.union({"init"}))) > 0:
                final.add(random.choice(list(states.difference(final.union({"init"})))))
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

    def get_random_accepted_words(self, nb):
        """
        Generate nb different accepted words
        :param nb: number of words to generate (int)
        :return: list of generated words (list)
        """
        reachable_states = self._compute_reachable_states(
            self.initial_state, self.input_symbols, self.transitions
        )
        if not any([state in self.final_states for state in reachable_states]):
            print("The language of this automaton is the empty gather.")
            return None
        regex = self.get_regex()
        if regex == "":
            return [""]
        if "*" not in regex and "+" not in regex:
            max_words = exrex.count(regex)
            if nb > max_words:
                nb = max_words
        list_words = list()
        same = 0
        while len(list_words) < nb:
            buffer = list_words.copy()
            generated_word = exrex.getone(regex, limit=30)
            if generated_word not in list_words:
                list_words.append(generated_word)
            if buffer == list_words:
                same += 1
            else:
                same = 0
            if same >= 100:
                break
        return list_words

    def classify_words(self, nb):
        """
        This function creates a 2 dimensional array with shape (np, 2) where the first column is a word in the alphabet
        and the second column is "1" if the word on the same row is accepted by the automaton and "0" otherwise
        :param nb: number of word in the array
        :return: 2 dimensional array with shape (np, 2) and datatype=str
        """
        classified = 0
        sigma_star = Automaton({0},
                               self.input_symbols,
                               {0: {char: {0} for char in self.input_symbols}},
                               0,
                               {0}).get_regex()
        classification = []
        for word in exrex.generate(sigma_star, limit=100):
            classification.append([word, int(self.accepts_input(word))])
            classified += 1
            if classified >= nb:
                return np.array(classification)

    def make_complete(self):
        """
        this function makes an equivalent complete NFA
        :return: None
        """
        states = set(self.states)
        states.add("well")
        transitions = dict()
        for key in self.transitions.keys():
            transitions[key] = {char: set(self.transitions[key][char]) for char in self.transitions[key].keys()}
        for state in states.difference({"init"}):
            if state not in transitions.keys():
                transitions[state] = {char: {"well"} for char in self.input_symbols}
            else:
                for char in self.input_symbols:
                    if char not in transitions[state].keys():
                        transitions[state][char] = {"well"}
        return Automaton(states, self.input_symbols, transitions, self.initial_state, self.final_states)

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
        alphabet, min_states, max_states, trans_density, init_amount, final_amount, init_range, final_range
    )
    print(automaton)
    
