from automaton_handler import Automaton
import numpy as np


def test_random_automaton():
    a = Automaton.random_automaton(
        alphabet={"0", "1"},
        min_states=2,
        max_states=10,
        trans_density=0.5,
        init_amount=1,
        final_amount=1,
        init_range=0,
        final_range=0,
        seed=None,
    )
    assert a.validate()


def test_classify_words():
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
    nb = 10
    classification = automaton.classify_words(nb)
    assert classification.shape == (nb, 2)
    assert classification.dtype.type == np.str_
    for row in classification:
        assert automaton.accepts_input(row[0]) == (row[1] == "1")
