import random
from automata.fa.nfa import NFA
from automata.fa.gnfa import GNFA
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class Automaton(NFA):
    def __init__(self, states, input_symbols, transitions, initial_state, final_states):
        super().__init__(states=states, input_symbols=input_symbols, transitions=transitions,
                         initial_state=initial_state, final_states=final_states)

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
        states = {'init'}
        for i in range(random.randint(min_states, max_states)):
            states.add(i)
        trans = {'init': {'': set()}}
        for state_in in states.difference({'init'}):
            for state_out in states.difference({'init'}):
                for char in alphabet:
                    if random.random() < trans_density:
                        try:
                            trans[state_in][char].add(state_out)
                        except KeyError:
                            try:
                                trans[state_in][char] = {state_out}
                            except KeyError:
                                trans[state_in] = {char: {state_out}}
        init = random.sample(states.difference({'init'}),
                             max(init_amount + random.randint(-init_range, init_range), 1))
        print(init)
        for i in init:
            trans['init'][''].add(i)
        final = set(random.sample(states.difference({'init'}),
                    final_amount + random.randint(-final_range, final_range)))
        return Automaton(states, alphabet, trans, 'init', final)

    def __str__(self):
        try:
            open('./display.png', "x")
        except FileExistsError:
            pass
        self.show_diagram(path='./display.png')
        plt.figure('Automate')
        plt.axis('off')
        plt.imshow(mpimg.imread('display.png'))
        plt.show()
        return super.__str__(self)

    def regex(self):
        return GNFA.from_nfa(self).to_regex()
