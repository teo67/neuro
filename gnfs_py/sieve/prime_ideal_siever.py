from sieve.lattice_siever import LatticeSiever
from sympy import poly
from sympy.abc import x as var_x
from sympy.polys import polytools
import utils
from net.neuron import Neuron
from net.net import Net
from collections.abc import Iterable
from typing import Any

class PrimeIdealSiever(LatticeSiever):
    coefficients: Iterable[int]
    d: int
    sympy_poly: Any
    B6: float

    def __init__(
        self, b_0: int, b_1: int, B: float, I: int, J: int, top_layer: Neuron, bottom_layer: Neuron, net: Net, coefficients: Iterable[int], B6: float
    ):
        self.coefficients = coefficients
        self.d = len(coefficients) - 1
        self.sympy_poly = 0
        self.B6 = B6
        for i, coeff in enumerate(self.coefficients):
            self.sympy_poly += (coeff * var_x ** i)
        super().__init__(b_0, b_1, B, I, J, top_layer, bottom_layer, net)

    def get_rp_pairs(self, b_0: int, b_1: int) -> Iterable[tuple[int, int]]:
        all_primes = utils.primesupto(b_1 + 1)
        return [(r, p) for p in all_primes for r in self.find_roots(p) if p >= b_0]

    def find_roots(self, prime: int) -> Iterable[int]:
        factors = polytools.factor_list(polytools.factor(self.sympy_poly, modulus=prime))[1]
        factor_nums = []
        for factor_p, one in factors:
            if one != 1:
                continue
            if polytools.degree(factor_p) != 1:
                continue
            factor_nums.append(-poly(factor_p).TC() % prime)
        return factor_nums

    def polynomial_f(self, x: float) -> float:
        return sum([coeff * (x**n) for n, coeff in enumerate(self.coefficients)])

    def get_norm(self, a: int, b: int) -> int:
        return round(pow(b, self.d) * self.polynomial_f(a/b))

    def get_threshold(self, u_1: int, u_2: int, v_1: int, v_2: int) -> float:
        return self.B6