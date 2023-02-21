from automaton_handler import Automaton
import tensorflow as ts
from tensorflow import keras


class Automaton2Network:
    def __init__(self, automaton):
        self.automaton = automaton
        self.network = None
