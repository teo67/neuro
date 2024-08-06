from neuron import Neuron
from synapse import Synapse
from net import Net
import numpy as np

def main():
    net = Net()
    n_1 = Neuron(
        voltage=np.array([0], dtype=int),
        leak_amount=np.array([1], dtype=int),
        threshold=np.array([10], dtype=int),
        resets_after_fire=np.array([True], dtype=bool)
    )
    n_2 = Neuron(
        voltage=np.array([0], dtype=int),
        leak_amount=np.array([0], dtype=int),
        threshold=np.array([2], dtype=int),
        resets_after_fire=np.array([True], dtype=bool)
    )
    net.register_neuron(n_1)
    net.register_neuron(n_2)
    s_1 = Synapse(
        n_1, n_2, np.array([[1]], dtype=int)
    )
    net.register_synapse(s_1)
    net.set_output_neuron(n_2)
    results = net.run_steps(100)
    print(results)

if __name__ == '__main__':
    main()