import numpy as np
from sieve.coprime_siever import CoprimeSiever
from sieve.prime_ideal_siever import PrimeIdealSiever
from sieve.prime_siever import PrimeSiever
from net.neuron import Neuron, make_neuron
from net.net import Net
from collections.abc import Iterable
import utils

class GNFSiever:
    net: Net
    top_layer: Neuron
    bottom_layer: Neuron
    coprime_siever: CoprimeSiever
    prime_siever: PrimeSiever
    prime_ideal_siever: PrimeIdealSiever
    I: int
    J: int

    def __init__(self, 
                 coefficients: Iterable[int], m: int, skew: int,
                 b_0_primes: int, b_1_primes: int, B_primes: float,
                 b_0_ideals: int, b_1_ideals: int, B_ideals: float, B6: float,
                 I: int, J: int):
        self.net = Net()
        self.top_layer = self.net.register_neuron(make_neuron(
            shape=(1,), voltage=1, threshold=J, leak_amount=1
        ))
        self.bottom_layer = self.net.register_neuron(make_neuron(
            shape=(1,), voltage=0, threshold=0, leak_amount=-2
        ))
        self.net.set_output_neuron(self.bottom_layer)
        self.coprime_siever = CoprimeSiever(I, J, self.top_layer, self.bottom_layer, self.net)
        self.prime_siever = PrimeSiever(b_0_primes, b_1_primes, B_primes, I, J, self.top_layer, self.bottom_layer, self.net, m)
        self.prime_ideal_siever = PrimeIdealSiever(b_0_ideals, b_1_ideals, B_ideals, I, J, self.top_layer, self.bottom_layer, self.net, coefficients, B6)
        self.I = I
        self.J = J
        self.skew = skew

    def find_basis(self, q: int, s: int) -> tuple[tuple[int, int], tuple[int, int]]:
        u = np.array([q, 0], dtype=float)
        v = np.array([s, self.skew], dtype=float)
        v_norm = np.linalg.norm(v)
        if v_norm > np.linalg.norm(u):
            temp = u
            u = v
            v = temp
            v_norm = q
        while v_norm <= np.linalg.norm(u):
            mult = round(np.dot(u, v) / (v_norm * v_norm))
            r = u - mult * v
            u = np.copy(v)
            v = r
            v_norm = np.linalg.norm(v)
        return (int(u[0]), int(u[1]/self.skew)), (int(v[0]), int(v[1]/self.skew))

    def update_neurons(self, u_1: int, u_2: int, v_1: int, v_2: int, debug: bool, verbose: bool):
        self.top_layer.set_voltage(1)
        self.coprime_siever.update_neurons()
        self.prime_siever.update_neurons(u_1, u_2, v_1, v_2, debug=debug, verbose=verbose)
        self.prime_ideal_siever.update_neurons(u_1, u_2, v_1, v_2, debug=debug, verbose=verbose)
        
    def sieve(self, q_min: int, q_max: int, debug: bool=False, verbose: bool=False) -> list[tuple[int, int, tuple[int, int], tuple[int, int], np.ndarray]]:
        print('Starting sieve, finding all q values...')
        all_qs = utils.primesupto(q_max + 1)
        chosen_qs = all_qs[all_qs >= q_min]
        print('done.')
        num_steps_per_q = self.I * self.J
        output = []
        for q in chosen_qs:
            for s in self.prime_ideal_siever.find_roots(q):
                if debug:
                    print(f'Now trying q={q}, s={s}')
                (u_1, u_2), (v_1, v_2) = self.find_basis(q, s)
                self.update_neurons(u_1, u_2, v_1, v_2, debug, verbose)
                results = self.net.run_steps(num_steps_per_q)
                if debug:
                    print('done')
                spike_times = np.where(results)[0]
                if verbose:
                    print(spike_times)
                if len(spike_times) > 0:
                    output.append((q, s, (u_1, u_2), (v_1, v_2), spike_times))
        print('Completed sieve.')
        return output
    
    def i_from_spike_time(self, t: int) -> int:
        return (t - 2)//self.J - self.I//2
    
    def j_from_spike_time(self, t: int) -> int:
        return (t - 1) % self.J
    
    def t_from_i_j(self, i: int, j: int) -> int:
        return (i + self.I//2) * self.J + j