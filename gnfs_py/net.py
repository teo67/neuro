from neuron import Neuron
from synapse import Synapse
import numpy as np

class Net:
    neurons: list[Neuron]
    synapses: list[Synapse]
    output_neuron: Neuron | None
    def __init__(self):
        self.neurons = []
        self.synapses = []
        self.output_neuron = None
    def register_neuron(self, neuron: Neuron):
        self.neurons.append(neuron)
    def register_synapse(self, synapse: Synapse):
        self.synapses.append(synapse)
    def set_output_neuron(self, output_neuron: Neuron):
        self.output_neuron = output_neuron
    def __run_single_step(self):
        for synapse in self.synapses:
            synapse.update()
        for neuron in self.neurons:
            neuron.run_step()
    def run_steps(self, num_steps: int) -> list[np.ndarray] | None:
        for neuron in self.neurons:
            neuron.reset_output()
        recorded_output = []
        for _ in range(num_steps):
            self.__run_single_step()
            if self.output_neuron is not None:
                recorded_output.append(self.output_neuron.get_last_output())
        return None if self.output_neuron is None else recorded_output
    