from net.neuron import Neuron, make_neuron
from net.synapse import Synapse, make_synapse
from net.net import Net
import numpy as np
from collections.abc import Iterable
import utils

class LatticeSiever:
    B: float
    I: int
    J: int
    r: np.ndarray
    p: np.ndarray
    primes_layer: Neuron
    local_bottom_layer: Neuron
    relay: Neuron
    top_to_primes: Synapse
    primes_to_primes: Synapse
    primes_to_local_bottom: Synapse
    top_to_relay: Synapse
    relay_to_local_bottom: Synapse
    local_bottom_to_bottom: Synapse

    def __init__(
        self, b_0: int, b_1: int, B: float, I: int, J: int, top_layer: Neuron, bottom_layer: Neuron, net: Net
    ):
        self.B = B
        self.I = I
        self.J = J
        rp_pairs = self.get_rp_pairs(b_0, b_1) # virtual function
        self.r = np.array([r for (r, p) in rp_pairs])
        self.p = np.array([p for (r, p) in rp_pairs])
        self.generate_neurons(top_layer, bottom_layer, net)
    def get_threshold(self, u_1: int, u_2: int, v_1: int, v_2: int) -> float: # virtual
        return 0

    def get_rp_pairs(self, b_0: int, b_1: int) -> Iterable[tuple[int, int]]: # virtual
        return ()

    def get_norm(self, a: int, b: int) -> int: # virtual
        return 0

    def check_any_factors(self, a: int, b: int) -> bool: 
        for r, p in zip(self.r, self.p):
            if (a - b * r) % p == 0:
                return True
        return False

    def get_reduced_norm(self, a: int, b: int) -> int:
        norm = self.get_norm(a, b)
        for r, p in zip(self.r, self.p):
            if (a - b * r) % p == 0:
                while norm % p == 0:
                    norm //= p
        return norm

    def check_smooth(self, a: int, b: int) -> bool: 
        return abs(self.get_reduced_norm(a, b)) == 1

    def generate_neurons(self, top_layer: Neuron, bottom_layer: Neuron, net: Net):
        num_p = len(self.p)
        
        self.primes_layer = net.register_neuron(make_neuron(
            shape=(num_p,), voltage=0, threshold=np.copy(self.p), leak_amount=1, resets_after_fire=False
        ))
        self.local_bottom_layer = net.register_neuron(make_neuron(
            shape=(1,), voltage=0, threshold=0, leak_amount=0, dtype=float
        ))
        self.relay = net.register_neuron(make_neuron(
            shape=(1,), voltage=0, threshold=0, leak_amount=-1
        ))

        self.top_to_primes = net.register_synapse(make_synapse(
            top_layer, self.primes_layer, 0
        ))
        primes_to_bottom_weights = [np.log(self.p)]
        self.primes_to_primes = net.register_synapse(make_synapse(self.primes_layer, self.primes_layer, -self.p))
        self.primes_to_local_bottom = net.register_synapse(make_synapse(self.primes_layer, self.local_bottom_layer, primes_to_bottom_weights))
        self.top_to_relay = net.register_synapse(make_synapse(top_layer, self.relay, 1))
        self.relay_to_local_bottom = net.register_synapse(make_synapse(self.relay, self.local_bottom_layer, -sum(primes_to_bottom_weights[0])))
        self.local_bottom_to_bottom = net.register_synapse(make_synapse(self.local_bottom_layer, bottom_layer, 1))
    def update_neurons(self, u_1: int, u_2: int, v_1: int, v_2: int, debug: bool=False, verbose: bool=False):
        # every q, we will need to update the primes layer and the top to prime synapse since these are dependent on q
        mod_inp = (v_1 - v_2 * self.r) % self.p
        which_special_cases = (mod_inp == 0)
        mod_inp[which_special_cases] = 1 # ignore these cases since the inverse is unsolvable
        if verbose:
            print(mod_inp, self.p)
            print('^ inputs to inv')
        
        modded = utils.mod_inv(mod_inp, self.p)
        if verbose:
            print(modded)
        ti = (u_1 - u_2 * self.r)
        if verbose:
            print(f'u={ti}')
            print(f'f={-(modded * ti)}')
        f = -(modded * ti) % self.p
        g_inner = -(f + self.J) % self.p
        g_inner[which_special_cases] = 0
        g = np.array([g_inner])
        V_0 = (self.I//2 * f)%self.p
        if verbose:
            print(f'V_0={V_0}, p={self.p}, special cases={which_special_cases}')
            print(f'g={g}')
        self.primes_layer.set_voltage(V_0)
        self.primes_layer.set_leak_amount(1 - which_special_cases)
        self.top_to_primes.set_weights(g.T)
        print(f'setting bottom leak amt to {self.B - self.get_threshold(u_1, u_2, v_1, v_2)}')
        self.local_bottom_layer.set_leak_amount(self.B - self.get_threshold(u_1, u_2, v_1, v_2))