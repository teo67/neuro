from sieve.lattice_siever import LatticeSiever
from sympy import poly, Poly
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
    sympy_poly: Poly
    B6: float

    def __init__(
        self, b_0: int, b_1: int, B: float, I: int, J: int, top_layer: Neuron, bottom_layer: Neuron, net: Net, coefficients: Iterable[int], B6: float
    ):
        """Create a prime ideal siever that fires when a(i, j) === rb(i, j) (mod p) and hook it up to the net.

        Args:
            b_0 (int): The lower bound (inclusive) for primes used in this sieve.
            b_1 (int): The upper bound (inclusive) for primes used in this sieve.
            B (float): The amount of lenience (increasing will cause this sieve to fire more).
            I (int): The (even) bound for the i coordinate (i <- [-I/2, I/2)).
            J (int): The bound for the j coordinate (j <- (0, J)).
            top_layer (Neuron): The row clock neuron (see GNFSiever) that provides input to the sieve.
            bottom_layer (Neuron): The final output neuron (see GNFSiever) that receives output from the sieve.
            net (Net): The net that this sieve is associated with.
            coefficients (Iterable[int]): The list of integer coefficients of the polynomial (index 0 = trailing coefficient, ...).
            B6 (float): The B6 value for this sieve (the constant smoothness threshold).
        """
        self.coefficients = coefficients
        self.d = len(coefficients) - 1
        self.sympy_poly = 0
        self.B6 = B6
        for i, coeff in enumerate(self.coefficients):
            self.sympy_poly += (coeff * var_x ** i)
        self.sympy_poly = Poly(self.sympy_poly, var_x)
        super().__init__(b_0, b_1, B, I, J, top_layer, bottom_layer, net)

    def get_rp_pairs(self, b_0: int, b_1: int) -> Iterable[tuple[int, int]]:
        all_primes = utils.primesupto(b_1 + 1)
        return [(r, p) for p in all_primes for r in self.find_roots(p) if p >= b_0]

    def find_roots(self, prime: int) -> Iterable[int]:
        """Find all roots of the polynomial associated with this sieve, mod a given prime.

        Args:
            prime (int): The prime p for which f(r) === 0, mod p.

        Returns:
            Iterable[int]: An iterable of integer roots, mod p.
        """
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
        """Get f(x), or the output of the polynomial associated with this sieve.

        Args:
            x (float): The input to the polynomial.

        Returns:
            float: The output of the polynomial.
        """
        return sum([coeff * (x**n) for n, coeff in enumerate(self.coefficients)])

    def get_norm(self, a: int, b: int) -> int:
        return round(pow(b, self.d) * self.polynomial_f(a/b))

    def get_threshold(self, u_1: int, u_2: int, v_1: int, v_2: int) -> float:
        return self.B6