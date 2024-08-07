import numpy as np
from typing import Any
import utils

class Neuron:
    voltage: np.ndarray
    leak_amount: np.ndarray
    threshold: np.ndarray
    resets_after_fire: np.ndarray
    last_output: np.ndarray
    incoming_synapse_amount: np.ndarray
    def __init__(self, 
                voltage: np.ndarray, leak_amount: np.ndarray,
                threshold: np.ndarray, resets_after_fire: np.ndarray):
        """Creates a new neuron layer with specified initial voltage, leak amount, threshold, and resetting behavior.

        Args:
            voltage (np.ndarray): The initial voltage of each neuron.
            leak_amount (np.ndarray): The leak amount (voltage += leak_amount at every timestep).
            threshold (np.ndarray): The voltage threshold that determines when the neurons will fire (voltage >= threshold).
            resets_after_fire (np.ndarray): Whether the neurons should reset their voltage sto 0 after they fire.
        """
        assert np.shape(voltage) == np.shape(leak_amount) == np.shape(threshold) == np.shape(resets_after_fire), \
            f'Neurons expect uniform shape but received shapes {np.shape(voltage)}, {np.shape(leak_amount)}, {np.shape(threshold)}, and {np.shape(resets_after_fire)}!'
        assert voltage.dtype == leak_amount.dtype == threshold.dtype, \
            f'Neurons expect uniform dtype but received types {voltage.dtype}, {leak_amount.dtype}, and {threshold.dtype}!'
        assert resets_after_fire.dtype == bool, \
            f'Neurons expect bool type for resets_after_fire but received f{resets_after_fire.dtype}!'
        self.voltage = voltage
        self.leak_amount = leak_amount
        self.threshold = threshold
        self.resets_after_fire = resets_after_fire
        self.reset_output()
        self.__reset_synapse_input()
    def run_step(self):
        """Runs a step of standard neuron behavior, which involves the following:\\
        V = V + Leak.\\
        V = V + Synapse Input.\\
        V >= Threshold --> Neuron Fires.\\
        Firing neurons that reset and neurons with negative voltage get set to 0.
        """
        self.voltage += self.leak_amount
        self.voltage += self.incoming_synapse_amount
        self.last_output = self.voltage >= self.threshold
        self.voltage[np.logical_or(
            np.logical_and(self.last_output, self.resets_after_fire), self.voltage < 0
        )] = 0
        self.__reset_synapse_input()
    def get_last_output(self) -> np.ndarray:
        """Get an array of bools that represents which neurons fired at the last timestep.

        Returns:
            np.ndarray: An array of bools where all True values correspond to neurons in this layer that fired.
        """
        return self.last_output
    def reset_output(self):
        """Clear the last firing output of this neuron.
        """
        self.last_output = np.zeros(np.shape(self.voltage), dtype=bool)
    def __reset_synapse_input(self):
        """(private method) Clear the synapse input of this neuron.
        """
        self.incoming_synapse_amount = np.zeros(np.shape(self.voltage), dtype=self.voltage.dtype)
    def get_shape(self) -> tuple[int]:
        """Get the numpy shape of this neuron layer, for example (2,) for a neuron with V = [3 4].

        Returns:
            tuple[int]: The shape of the state of this neuron layer.
        """
        return np.shape(self.voltage)
    def get_dtype(self) -> Any:
        """Get the numpy datatype of this neuron layer, for example float for a neuron with V = [3. 4.].

        Returns:
            Any: The type of data stored in this neuron layer's state.
        """
        return self.voltage.dtype
    def add_synapse_input(self, amount: np.ndarray):
        """Give this neuron a certain amount of synapse input, which it will add to its voltage at the next timestep.

        Args:
            amount (np.ndarray): An array that matches the shape and datatype of this neuron.
        """
        self.incoming_synapse_amount += amount
    def __check_update_property(self, incoming: Any, old_value: np.ndarray, name: str) -> np.ndarray:
        """(private method) Transform incoming data into a format that fits the neuron and validate its format.

        Args:
            incoming (Any): The data to be transformed, which can be a single value or list/tuple/array.
            old_value (np.ndarray): The previous value of the property that is being updated.
            name (str): The name of the property being updated, for debugging purposes.

        Returns:
            np.ndarray: The transformed value of the data that was passed in.
        """
        array_incoming = utils.get_np_array(incoming, np.shape(old_value), old_value.dtype)
        utils.check_update_property(array_incoming, old_value, f'neuron.{name}')
        return array_incoming
    def set_voltage(self, voltage: Any):
        """Set the voltage of this neuron layer to a new value.

        Args:
            voltage (Any): A single value or list/tuple/array (must fit the shape of this neuron) representing voltage.
        """
        self.voltage = self.__check_update_property(voltage, self.voltage, 'voltage')
    def set_leak_amount(self, leak_amount: Any):
        """Set the leak amount of this neuron layer to a new value.

        Args:
            leak_amount (Any): A single value or list/tuple/array (must fit the shape of this neuron) representing leak amount.
        """
        self.leak_amount = self.__check_update_property(leak_amount, self.leak_amount, 'leak amount')
    def set_threshold(self, threshold: Any):
        """Set the threshold of this neuron layer to a new value.

        Args:
            threshold (Any): A single value or list/tuple/array (must fit the shape of this neuron) representing threshold.
        """
        self.threshold = self.__check_update_property(threshold, self.threshold, 'threshold')
    def set_resets_after_fire(self, resets_after_fire: Any):
        """Set the resetting behavior of this neuron layer to a new value.

        Args:
            resets_after_fire (Any): A single bool or list/tuple/array of bools (must fit the shape of this neuron) representing resetting behavior.
        """
        self.resets_after_fire = self.__check_update_property(resets_after_fire, self.resets_after_fire, 'resets after fire')
    def __str__(self) -> str:
        """Get a string representing the state of this neuron.

        Returns:
            str: A string containing this neuron's voltage, threshold, leak amount, and resetting behavior.
        """
        return f'Neuron[voltage={self.voltage}, threshold={self.threshold}, leak={self.leak_amount}, reset={self.resets_after_fire}]'
def make_neuron(
        shape: tuple[int], voltage: Any, leak_amount: Any, threshold: Any, resets_after_fire: Any = True, dtype: Any = int
) -> Neuron:
    """A helper function to create a neuron more easily, with more flexible inputs.

    Args:
        shape (tuple[int]): The desired shape of the new neuron.
        voltage (Any): The voltage of the new neuron, either a single value or a list/tuple/array that fits the desired shape.
        leak_amount (Any): The leak amount of the new neuron, either a single value or a list/tuple/array that fits the desired shape.
        threshold (Any): The threshold of the new neuron, either a single value or a list/tuple/array that fits the desired shape.
        resets_after_fire (Any, optional): The resetting behavior of the new neuron, either a single bool or a list/tuple/array of bools that fits the desired shape. Defaults to True.
        dtype (Any, optional): The desired datatype of the new neuron. Defaults to int.

    Returns:
        Neuron: A neuron object storing the values provided.
    """
    return Neuron(
        voltage=utils.get_np_array(voltage, shape, dtype),
        leak_amount=utils.get_np_array(leak_amount, shape, dtype),
        threshold=utils.get_np_array(threshold, shape, dtype),
        resets_after_fire=utils.get_np_array(resets_after_fire, shape, bool)
    )