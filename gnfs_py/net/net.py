from net.neuron import Neuron, make_neuron
from net.synapse import Synapse, make_synapse
import numpy as np
from typing import Any

class Net:
    neurons: list[Neuron]
    synapses: list[Synapse]
    output_neuron: Neuron | None
    def __init__(self):
        self.neurons = []
        self.synapses = []
        self.output_neuron = None
    def register_neuron(self, neuron: Neuron) -> Neuron:
        self.neurons.append(neuron)
        return neuron
    def register_synapse(self, synapse: Synapse) -> Synapse:
        self.synapses.append(synapse)
        return synapse
    def set_output_neuron(self, output_neuron: Neuron):
        self.output_neuron = output_neuron
    def __run_single_step(self):
        for synapse in self.synapses:
            synapse.update()
        for neuron in self.neurons:
            neuron.run_step()
    def run_steps(self, num_steps: int, carryover_output: bool = False) -> list[np.ndarray] | None:
        if not carryover_output:
            for neuron in self.neurons:
                neuron.reset_output()
        recorded_output = []
        for _ in range(num_steps):
            self.__run_single_step()
            if self.output_neuron is not None:
                recorded_output.append(self.output_neuron.get_last_output())
        return None if self.output_neuron is None else recorded_output
    