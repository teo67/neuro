from neuron import Neuron
import numpy as np

class Synapse:
    n_from: Neuron
    n_to: Neuron
    weights: np.ndarray
    def __init__(self, n_from: Neuron, n_to: Neuron, weights: np.ndarray):
        assert np.shape(weights) == n_to.get_shape() + n_from.get_shape(), \
            f'Synapse bridging neuron of shape {n_from.get_shape()} to neuron of shape {n_to.get_shape()} has invalid shape {np.shape(weights)}!'
        assert weights.dtype == n_to.get_dtype(), \
            f'Synapse towards neuron of type {n_to.get_dtype()} was given weights of type {weights.dtype}!'
        self.n_from = n_from
        self.n_to = n_to
        self.weights = weights
    def update(self):
        self.n_to.add_synapse_input(self.weights @ self.n_from.get_last_output())