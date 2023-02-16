import random


class Automaton:
    def __init__(self, alphabet, states, trans, init, final):
        self.alphabet = alphabet
        self.states = states
        self.trans = trans
        self.init = init
        self.final = final

    @staticmethod
    def random_automaton(alphabet, min_states, max_states, trans_density, init_amount, final_amount, init_range=0,
                         final_range=0, seed=None):
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
        states = set()
        for i in range(random.randint(min_states, max_states)):
            states.add(i)
        trans = dict()
        for state_in in states:
            for state_out in states:
                for char in alphabet:
                    if random.random() < trans_density:
                        try:
                            trans[state_in][char].add(state_out)
                        except KeyError:
                            trans[state_in] = {char: {state_out}}
        init = set(random.sample(states, init_amount + random.randint(-init_range, init_range)))
        final = set(random.sample(states, final_amount + random.randint(-final_range, final_range)))
        return states, trans, init, final
