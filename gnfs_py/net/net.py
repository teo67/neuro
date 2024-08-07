from net.neuron import Neuron, make_neuron
from net.synapse import Synapse, make_synapse
import numpy as np
from typing import Any

class Net:
    neurons: list[Neuron]
    synapses: list[Synapse]
    output_neuron: Neuron | None
    def __init__(self):
        """Create a blank net with no neurons or synapses.
        """
        self.neurons = []
        self.synapses = []
        self.output_neuron = None
    def register_neuron(self, neuron: Neuron) -> Neuron:
        """Add a neuron object to the net, so that it will run when the net is running.

        Args:
            neuron (Neuron): The neuron to be added to the net's register.

        Returns:
            Neuron: The neuron that was passed in.
        """
        self.neurons.append(neuron)
        return neuron
    def register_synapse(self, synapse: Synapse) -> Synapse:
        """Add a synapse object to the net, so that it will run when the net is running.

        Args:
            synapse (Synapse): The synapse to be added to the net's register.

        Returns:
            Synapse: The synapse that was passed in.
        """
        self.synapses.append(synapse)
        return synapse
    def set_output_neuron(self, output_neuron: Neuron):
        """Change the output neuron of this net (see Net.run_steps).

        Args:
            output_neuron (Neuron): The new output neuron for this net.
        """
        self.output_neuron = output_neuron
    def __run_single_step(self):
        """(private method) Run a single step of the net without recording output.
        """
        for synapse in self.synapses:
            synapse.update()
        for neuron in self.neurons:
            neuron.run_step()
    def run_steps(self, num_steps: int, carryover_output: bool = False) -> list[np.ndarray] | None:
        """Run the net for a given number of steps. The running process for each step is as follows:\\
        All synapses will calculate their output and feed it to their output neurons.\\
        All neurons will run, updating their voltage and sometimes firing.\\
        If this net has an output neuron, its firing behavior will be recorded.

        Args:
            num_steps (int): The number of steps to run the net for.
            carryover_output (bool, optional): If set to True, neuron outputs from the last step before calling this function
            will not be cleared. Useful for running a net one step at a time to do debugging. Defaults to False.

        Returns:
            list[np.ndarray] | None: A list of spike outputs for this net's output neuron, if there is one. Otherwise, None.
        """
        if not carryover_output:
            for neuron in self.neurons:
                neuron.reset_output()
        recorded_output = []
        for _ in range(num_steps):
            self.__run_single_step()
            if self.output_neuron is not None:
                recorded_output.append(self.output_neuron.get_last_output())
        return None if self.output_neuron is None else recorded_output
    