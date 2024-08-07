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
        self.voltage += self.leak_amount
        self.voltage += self.incoming_synapse_amount
        self.last_output = self.voltage >= self.threshold
        self.voltage[np.logical_or(
            np.logical_and(self.last_output, self.resets_after_fire), self.voltage < 0
        )] = 0
        self.__reset_synapse_input()
    def get_last_output(self) -> np.ndarray:
        return self.last_output
    def reset_output(self):
        self.last_output = np.zeros(np.shape(self.voltage), dtype=bool)
    def __reset_synapse_input(self):
        self.incoming_synapse_amount = np.zeros(np.shape(self.voltage), dtype=self.voltage.dtype)
    def get_shape(self) -> tuple[int]:
        return np.shape(self.voltage)
    def get_dtype(self) -> Any:
        return self.voltage.dtype
    def add_synapse_input(self, amount: np.ndarray):
        self.incoming_synapse_amount += amount
    def __check_update_property(self, incoming: Any, old_value: np.ndarray, name: str) -> np.ndarray:
        array_incoming = utils.get_np_array(incoming, np.shape(old_value), old_value.dtype)
        utils.check_update_property(array_incoming, old_value, f'neuron.{name}')
        return array_incoming
    def set_voltage(self, voltage: Any):
        self.voltage = self.__check_update_property(voltage, self.voltage, 'voltage')
    def set_leak_amount(self, leak_amount: Any):
        self.leak_amount = self.__check_update_property(leak_amount, self.leak_amount, 'leak amount')
    def set_threshold(self, threshold: Any):
        self.threshold = self.__check_update_property(threshold, self.threshold, 'threshold')
    def set_resets_after_fire(self, resets_after_fire: Any):
        self.resets_after_fire = self.__check_update_property(resets_after_fire, self.resets_after_fire, 'resets after fire')
    def __str__(self):
        return f'Neuron[voltage={self.voltage}, threshold={self.threshold}, leak={self.leak_amount}, reset={self.resets_after_fire}]'
def make_neuron(
        shape: tuple[int], voltage: Any, leak_amount: Any, threshold: Any, resets_after_fire: Any = True, dtype: Any = int
) -> Neuron:
    return Neuron(
        voltage=utils.get_np_array(voltage, shape, dtype),
        leak_amount=utils.get_np_array(leak_amount, shape, dtype),
        threshold=utils.get_np_array(threshold, shape, dtype),
        resets_after_fire=utils.get_np_array(resets_after_fire, shape, bool)
    )