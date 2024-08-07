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
        """Create a synapse between two neurons with a certain matrix of weights.

        Args:
            n_from (Neuron): The neuron that provides input to this synapse.
            n_to (Neuron): The neuron that receives output from this synapse.
            weights (np.ndarray): An array/matrix of weights to transform boolean input to numeric output.
            If the shape of the weights is such that weights @ input = output, then the synapse will behave that way.
            Otherwise, if the shape of the weights matches both the shape of the input and the shape of the output, 
            the output will be determined by pointwise multiplication of the weights and the input.
        """
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
        """Calculate synapse output and feed it to the receiving neuron.
        """
        self.n_to.add_synapse_input(
            np.multiply(self.weights, self.n_from.get_last_output())
            if self.is_pointwise else
            self.weights @ self.n_from.get_last_output())
    def set_weights(self, weights: Any):
        """Change the weights of this synapse.

        Args:
            weights (Any): A single value or list/tuple/array that matches the shape of this synapse.
        """
        weights_array = utils.get_np_array(weights, np.shape(self.weights), self.weights.dtype)
        utils.check_update_property(weights_array, self.weights, 'synapse.weights')
        self.weights = weights_array
    def __str__(self) -> str:
        """Get a string representing the state of the neurons enclosed in this synapse as well as the synapse weights.

        Returns:
            str: A string with the state of both neurons as well as the synapse weights.
        """
        return f'Synapse[from {self.n_from} to {self.n_to}, weights: {self.weights} {"(ptwise)" if self.is_pointwise else ""}]'
def make_synapse(
        n_from: Neuron, n_to: Neuron, weights: Any, is_pointwise: bool = False
) -> Synapse:
    """A helper function to create synapses with more flexible inputs.

    Args:
        n_from (Neuron): The neuron which should send input to this synapse.
        n_to (Neuron): The neuron which should receive output from this synapse.
        weights (Any): A single value or list/tuple/array of weights.
        is_pointwise (bool, optional): If a single value is passed in, this determines whether the synapse 
        should do matrix or pointwise multiplication (if a list/tuple/array is passed in, that can be determined automatically). 
        Defaults to False.

    Returns:
        Synapse: A synapse object storing the values provided.
    """
    if type(weights) == list or type(weights) == tuple or type(weights) == np.ndarray:
        weights_array = np.array(weights, dtype=n_to.get_dtype())
    else: 
        weight_shape = n_from.get_shape() if is_pointwise else (n_to.get_shape() + n_from.get_shape())
        weights_array = np.full(weight_shape, weights, dtype=n_to.get_dtype())
    return Synapse(n_from, n_to, weights_array)