import utils
from net.net import Net
from net.neuron import Neuron, make_neuron
from net.synapse import Synapse, make_synapse
import numpy as np

class CoprimeSiever:
    I: int
    J: int
    primes_using: np.ndarray
    i_primes_layer: Neuron
    j_primes_layer: Neuron
    izero_layer: Neuron
    local_bottom_layer: Neuron
    top_to_i_primes: Synapse
    top_to_j_primes: Synapse
    i_primes_to_j_primes: Synapse
    j_primes_to_j_primes: Synapse
    top_to_izero: Synapse
    j_primes_to_local_bottom: Synapse
    izero_to_izero: Synapse
    izero_to_local_bottom: Synapse
    local_bottom_to_bottom: Synapse

    def __init__(self, I: int, J: int, top_layer: Neuron, bottom_layer: Neuron, net: Net):
        self.I = I
        self.J = J
        self.generate_neurons(top_layer, bottom_layer, net)
    def generate_neurons(self, top_layer: Neuron, bottom_layer: Neuron, net: Net):
        self.primes_using = utils.primesupto(min(self.I//2+1, self.J))
        num_p = len(self.primes_using)
        self.i_primes_layer = net.register_neuron(make_neuron(
            shape=(num_p,),
            voltage=0,
            threshold=np.copy(self.primes_using), # don't want to pass this in by reference
            leak_amount=0
        ))
        self.j_primes_layer = net.register_neuron(make_neuron(
            shape=(num_p,),
            voltage=0,
            threshold=self.primes_using + self.J,
            leak_amount=1,
            resets_after_fire=False
        ))
        self.local_bottom_layer = net.register_neuron(make_neuron(
            shape=(1,),
            voltage=0, threshold=0, leak_amount=-1
        ))
        self.izero_layer = net.register_neuron(make_neuron(
            shape=(2,),
            voltage=0, threshold=[self.I//2, 1], leak_amount=0, resets_after_fire=[True, False]
        ))
        self.top_to_i_primes = net.register_synapse(make_synapse(top_layer, self.i_primes_layer, np.ones((num_p, 1))))
        self.top_to_j_primes = net.register_synapse(make_synapse(top_layer, self.j_primes_layer, [
            [-(self.J + p)] for p in self.primes_using
        ]))
        self.i_primes_to_j_primes = net.register_synapse(make_synapse(self.i_primes_layer, self.j_primes_layer,
            [self.J for _ in range(num_p)]
        ))
        self.j_primes_to_j_primes = net.register_synapse(make_synapse(self.j_primes_layer, self.j_primes_layer, 
            -self.primes_using
        ))
        self.top_to_izero = net.register_synapse(make_synapse(top_layer, self.izero_layer, [[1], [-1]]))
        self.j_primes_to_local_bottom = net.register_synapse(make_synapse(self.j_primes_layer, self.local_bottom_layer, np.ones((1, num_p))))
        self.izero_to_izero = net.register_synapse(make_synapse(self.izero_layer, self.izero_layer, [[0, 0], [1, 0]]))
        self.izero_to_local_bottom = net.register_synapse(make_synapse(self.izero_layer, self.local_bottom_layer, [[0, 1]]))
        self.local_bottom_to_bottom = net.register_synapse(make_synapse(self.local_bottom_layer, bottom_layer, -1))
    def update_neurons(self):
        i_0_mod_p = (-self.I//2)%self.primes_using
        self.i_primes_layer.set_voltage(i_0_mod_p)
        self.j_primes_layer.set_voltage(np.where(i_0_mod_p == 0, self.J, 0))
        self.izero_layer.set_voltage(0)