import numpy as np
from typing import Any

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