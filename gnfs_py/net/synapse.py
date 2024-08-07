from net.neuron import Neuron
import numpy as np
import utils 
from typing import Any

class Synapse:
    n_from: Neuron
    n_to: Neuron
    weights: np.ndarray
    is_pointwise: bool
    def __init__(self, n_from: Neuron, n_to: Neuron, weights: np.ndarray):
        if np.shape(weights) == n_to.get_shape() + n_from.get_shape():
            self.is_pointwise = False
        else:
            self.is_pointwise = True
            assert n_from.get_shape() == n_to.get_shape() == np.shape(weights), \
                f'Synapse bridging neuron of shape {n_from.get_shape()} to neuron of shape {n_to.get_shape()} has invalid shape {np.shape(weights)}!'
        assert weights.dtype == n_to.get_dtype(), \
            f'Synapse towards neuron of type {n_to.get_dtype()} was given weights of type {weights.dtype}!'
        self.n_from = n_from
        self.n_to = n_to
        self.weights = weights
    def update(self):
        self.n_to.add_synapse_input(
            np.multiply(self.weights, self.n_from.get_last_output())
            if self.is_pointwise else
            self.weights @ self.n_from.get_last_output())
    def set_weights(self, weights: Any):
        weights_array = utils.get_np_array(weights, np.shape(self.weights), self.weights.dtype)
        utils.check_update_property(weights_array, self.weights, 'synapse.weights')
        self.weights = weights_array
    def __str__(self):
        return f'Synapse[from {self.n_from} to {self.n_to}, weights: {self.weights} {"(ptwise)" if self.is_pointwise else ""}]'
def make_synapse(
        n_from: Neuron, n_to: Neuron, weights: Any, is_pointwise: bool = False
) -> Synapse:
    if type(weights) == list or type(weights) == tuple or type(weights) == np.ndarray:
        weights_array = np.array(weights, dtype=n_to.get_dtype())
    else: 
        weight_shape = n_from.get_shape() if is_pointwise else (n_to.get_shape() + n_from.get_shape())
        weights_array = np.full(weight_shape, weights, dtype=n_to.get_dtype())
    return Synapse(n_from, n_to, weights_array)