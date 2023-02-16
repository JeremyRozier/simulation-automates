import random


class Automaton:
    def __init__(self, alphabet, states, trans, init, final):
        self.alphabet = alphabet
        self.states = states
        self.trans = trans
        self.init = init
        self.final = final

    @staticmethod
    def random_automaton(alphabet, min_states, max_states, trans_density, init_density, final_density, seed=None):
        """
        This function creates a random automaton with desired probabilities

        :param alphabet: set of the alphabet used by the automaton (set of str)
        :param min_states: minimum amount of states (int)
        :param max_states: maximum amount of states (int)
        :param trans_density: chance for a transition to be created (percentage)
        :param init_density: chance for a state to become initial (percentage)
        :param final_density: chance for a state to become final (percentage)
        :param seed: random seed, None by default
        :return: a set of states, a dictionary of transition, set of initial states, set of final states
        """
        if seed is not None:
            random.seed(seed)
        states, init, final = set(), set(), set()
        for i in range(random.randint(min_states, max_states)):
            states.add(i)
            if random.random() < init_density:
                init.add(i)
            if random.random() < final_density:
                final.add(i)
        trans = dict()
        for state_in in states:
            for state_out in states:
                for char in alphabet:
                    if random.random() < trans_density:
                        try:
                            trans[state_in][char].add(state_out)
                        except KeyError:
                            trans[state_in] = {char: {state_out}}
        return states, trans, init, final
