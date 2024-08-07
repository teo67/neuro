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
        """Create a lattice siever that fires when a(i, j) === rb(i, j) (mod p) and hook it up to the net.

        Args:
            b_0 (int): The lower bound (inclusive) for primes used in this sieve.
            b_1 (int): The upper bound (inclusive) for primes used in this sieve.
            B (float): The amount of lenience (increasing will cause this sieve to fire more).
            I (int): The (even) bound for the i coordinate (i <- [-I/2, I/2)).
            J (int): The bound for the j coordinate (j <- (0, J)).
            top_layer (Neuron): The row clock neuron (see GNFSiever) that provides input to the sieve.
            bottom_layer (Neuron): The final output neuron (see GNFSiever) that receives output from the sieve.
            net (Net): The net that this sieve is associated with.
        """
        assert I%2==0, "I must be even!"
        self.B = B
        self.I = I
        self.J = J
        rp_pairs = self.get_rp_pairs(b_0, b_1) # virtual function
        self.r = np.array([r for (r, p) in rp_pairs])
        self.p = np.array([p for (r, p) in rp_pairs])
        self.__generate_neurons(top_layer, bottom_layer, net)
    def get_threshold(self, u_1: int, u_2: int, v_1: int, v_2: int) -> float: # virtual
        """A virtual method to get the smoothness threshold for this sieve based on the q, s basis that it is about to run.

        Args:
            u_1 (int): The first coordinate of the u vector.
            u_2 (int): The second coordinate of the u vector.
            v_1 (int): The first coordinate of the v vector.
            v_2 (int): The second coordinate of the v vector.

        Returns:
            float: The smoothness threshold corresponding to the basis.
        """
        return 0

    def get_rp_pairs(self, b_0: int, b_1: int) -> Iterable[tuple[int, int]]: # virtual
        """A virtual method to get the (root, prime) pairs for this sieve based on the prime range of [b_0, b_1].

        Args:
            b_0 (int): The inclusive lower bound for generated primes.
            b_1 (int): The inclusive upper bound for generated primes.

        Returns:
            Iterable[tuple[int, int]]: An iterable of (root, prime) tuples for this sieve's prime base.
        """
        return ()

    def get_norm(self, a: int, b: int) -> int: # virtual
        """A virtual method to get the norm associated with this sieve based on an (a, b) pair.

        Args:
            a (int): The a value.
            b (int): The b value.

        Returns:
            int: The norm of (a, b) (can be negative).
        """
        return 0

    def check_any_factors(self, a: int, b: int) -> bool: 
        """Determine whether (a, b) has any factors in this sieve's prime base.

        Args:
            a (int): The a value.
            b (int): The b value.

        Returns:
            bool: True if there is at least one (root, prime) pair in the prime base that satisfies a === rb (mod p).
        """
        for r, p in zip(self.r, self.p):
            if (a - b * r) % p == 0:
                return True
        return False

    def get_reduced_norm(self, a: int, b: int) -> int:
        """Get the norm of (a, b) and then divide it by primes in the prime base as much as possible.

        Args:
            a (int): The a value.
            b (int): The b value.

        Returns:
            int: The result of dividing the norm by primes as many times as possible (can be negative).
        """
        norm = self.get_norm(a, b)
        for r, p in zip(self.r, self.p):
            if (a - b * r) % p == 0:
                while norm % p == 0:
                    norm //= p
        return norm

    def check_smooth(self, a: int, b: int) -> bool: 
        """Check whether a pair (a, b) is completely smooth in the context of this sieve's prime base.

        Args:
            a (int): The a value.
            b (int): The b value.

        Returns:
            bool: True if (a, b) is smooth.
        """
        return abs(self.get_reduced_norm(a, b)) == 1

    def __generate_neurons(self, top_layer: Neuron, bottom_layer: Neuron, net: Net):
        """(private method) Generates all neurons and synapses used for this sieve, and registers them all to the net.

        Args:
            top_layer (Neuron): The row clock neuron.
            bottom_layer (Neuron): The final output neuron that this sieve should fire towards.
            net (Net): The net to register neurons and synapses to.
        """
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
        """Reset neuron values and synapse weights to prepare for running a new q-value, based on the q, s basis that was determined.

        Args:
            u_1 (int): The first coordinate of the u vector.
            u_2 (int): The second coordinate of the u vector.
            v_1 (int): The first coordinate of the v vector.
            v_2 (int): The second coordinate of the v vector.
            debug (bool, optional): Whether debug strings should be printed. Defaults to False.
            verbose (bool, optional): Whether verbose strings should be printed. Defaults to False.
        """
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
        self.local_bottom_layer.set_leak_amount(min(self.B - self.get_threshold(u_1, u_2, v_1, v_2), -0.01)) # we want this to be at least negative