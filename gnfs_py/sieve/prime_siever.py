from sieve.lattice_siever import LatticeSiever
import utils
from net.neuron import Neuron
from net.net import Net
from collections.abc import Iterable
import numpy as np

class PrimeSiever(LatticeSiever):
    m: int

    def __init__(
        self, b_0: int, b_1: int, B: float, I: int, J: int, top_layer: Neuron, bottom_layer: Neuron, net: Net, m: int
    ):
        self.m = m
        super().__init__(b_0, b_1, B, I, J, top_layer, bottom_layer, net)

    def get_rp_pairs(self, b_0: int, b_1: int) -> Iterable[tuple[int, int]]:
        all_primes = utils.primesupto(b_1 + 1)
        return [(self.m, p) for p in all_primes if p >= b_0]

    def get_norm(self, a: int, b: int) -> int:
        return a - b * self.m

    def get_threshold(self, u_1: int, u_2: int, v_1: int, v_2: int) -> float:
        diff_1: int = u_1 - self.m * u_2
        diff_2: int = v_1 - self.m * v_2
        mean_square: float = (self.J - 1)/(12*self.J) * (self.I*self.I + 2) * (diff_1*diff_1) \
                    + (1/6) * (self.J - 1) * (2*self.J - 1) * (diff_2*diff_2) \
                    - (1/2) * (self.J - 1) * diff_1 * diff_2 # this was derived from avg((a - bm)^2)
        root_mean_square: float = pow(mean_square, 0.5)
        return np.log(root_mean_square) # we want numbers that are smoother than the magnitude (or RMS) of the average