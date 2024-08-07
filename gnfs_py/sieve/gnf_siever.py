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
        """Create a General Number Field Siever that fires when (i, j) pairs are smooth in both domains and coprime.

        Args:
            coefficients (Iterable[int]): The integer coefficients of the polynomial to be used in the sieve.
            m (int): The m value to be used in the sieve.
            skew (int): The skew, which determines the relative size of components in the u and v vectors.
            b_0_primes (int): The (inclusive) lower bound for primes in the prime sieve.
            b_1_primes (int): The (inclusive) upper bound for primes in the prime sieve.
            B_primes (float): The lenience amount for smoothness in the primes.
            b_0_ideals (int): The (inclusive) lower bound for primes in the prime ideal sieve.
            b_1_ideals (int): The (inclusive) upper bound for primes in the prime ideal sieve.
            B_ideals (float): The lenience amount for smoothness in the prime ideals.
            B6 (float): The constant threshold value for the prime ideal sieve.
            I (int): The (even) bound for i values (i <- [-I/2, I/2)).
            J (int): The bound for j values (j <- (0, J)).
        """
        assert I%2==0, "I must be even!"
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
        """Given a special q value and a root s, find a basis (u, v) of the space that satisfies a === sb (mod q).

        Args:
            q (int): The special q value (prime).
            s (int): The root, such that f(s) === 0 (mod q).

        Returns:
            tuple[tuple[int, int], tuple[int, int]]: (u1, u2), (v1, v2), where u and v are the two basis vectors.
        """
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
        """Update all neurons and synapses in this sieve and inner sieves to prepare for a new q value.

        Args:
            u_1 (int): The first coordinate of the u vector.
            u_2 (int): The second coordinate of the u vector.
            v_1 (int): The first coordinate of the v vector.
            v_2 (int): The second coordinate of the v vector.
            debug (bool): Whether to print debug strings.
            verbose (bool): Whether to print verbose strings.
        """
        self.top_layer.set_voltage(1)
        self.coprime_siever.update_neurons()
        self.prime_siever.update_neurons(u_1, u_2, v_1, v_2, debug=debug, verbose=verbose)
        self.prime_ideal_siever.update_neurons(u_1, u_2, v_1, v_2, debug=debug, verbose=verbose)
        
    def sieve(self, q_min: int, q_max: int, debug: bool=False, verbose: bool=False) -> list[tuple[int, int, tuple[int, int], tuple[int, int], np.ndarray]]:
        """Run the General Number Field Sieve for a given range of q values and return the results.

        Args:
            q_min (int): The (inclusive) minimum value for special qs.
            q_max (int): The (inclusive) maximum value for special qs.
            debug (bool, optional): Whether to print debug strings. Defaults to False.
            verbose (bool, optional): Whether to print verbose strings. Defaults to False.

        Returns:
            list[tuple[int, int, tuple[int, int], tuple[int, int], np.ndarray]]: The output, formatted as a list of
            (q, s, (u1, u2), (v1, v2), [spike times]) tuples for each (q, s) pair that led to spikes.
        """
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
        """Work backwards to find the i value from a given spike time.

        Args:
            t (int): The spike time.

        Returns:
            int: The i value.
        """
        return (t - 2)//self.J - self.I//2
    
    def j_from_spike_time(self, t: int) -> int:
        """Work backwards to find the j value from a given spike time.

        Args:
            t (int): The spike time.

        Returns:
            int: The j value.
        """
        return (t - 1) % self.J
    
    def t_from_i_j(self, i: int, j: int) -> int:
        """Given an (i, j) pair, determine the time at which a spike would occur.

        Args:
            i (int): The i value.
            j (int): The j value.

        Returns:
            int: The spike time.
        """
        return (i + self.I//2) * self.J + j