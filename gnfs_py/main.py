from sieve.gnf_siever import GNFSiever
from sieve.post_processor import PostProcessor
import numpy as np
import galois
import utils
from sympy import poly, Poly, diff, ZZ
from sympy.abc import x as var_x
from sympy.polys import polytools
# import utils
# from net.neuron import Neuron, make_neuron
# from net.net import Net
# from net.synapse import Synapse, make_synapse

def main():
    n = 45113
    prime_upper_bound = 29 # B_0 in Lattice Sieve paper
    ideals_upper_bound = 103 # B_1 in Lattice Sieve paper
    large_prime_upper_bound = 1000 # B_2 in Lattice Sieve paper
    siever = GNFSiever(
        coefficients=(8, 29, 15, 1), m=31, skew=1,
        b_0_primes=1, b_1_primes=prime_upper_bound, B_primes=np.log(large_prime_upper_bound), # so as to allow up to one large prime
        b_0_ideals=1, b_1_ideals=ideals_upper_bound, B_ideals=np.log(large_prime_upper_bound), B6=10,
        I=20, J=10)
    
    # my_poly = Poly(58251363820606365*var_x**2+149816899035790332*var_x+75158930297695972, var_x)
    
    # print(utils.poly_sqrt(my_poly, Poly(siever.prime_ideal_siever.sympy_poly), siever.prime_ideal_siever.p))
    # return
    
    results = siever.sieve(prime_upper_bound, ideals_upper_bound, debug=False, verbose=False)
    post_processor = PostProcessor(siever, results)
    print()
    print('Raw results')
    print(results)
    print()
    print('Filtering...')
    res = post_processor.check_results()
    if res is None:
        print('Checking process halted early.')
        return
    smooth_ab, nonsmooth_ab = res
    print()
    print('Smooth results')
    print(smooth_ab)
    print()
    print(f'Found {len(smooth_ab)} smooth (a, b) pairs.')
    print()
    print('Nonsmooth results')
    print(nonsmooth_ab)
    print()
    print(f'Found {len(nonsmooth_ab)} non-smooth (a, b) pairs.')
    print()
    print('Finding extra smooth pairs...')
    extra_smooth_ab, extra_prime_factors, extra_prime_ideal_factors = post_processor.find_extra_smooth_pairs()
    post_processor.validate_extra_smooth_pairs()
    print()
    print('Extra smooth pairs')
    print(extra_smooth_ab)
    print()
    print('Extra prime factors')
    print(extra_prime_factors)
    print()
    print('Extra prime ideal factors')
    print(extra_prime_ideal_factors)
    print()
    print(f'Found {len(extra_smooth_ab)} extra smooth pairs, using {len(extra_prime_factors)} extra prime factors and {len(extra_prime_ideal_factors)} extra prime ideal factors.')
    diff = len(extra_smooth_ab) - len(extra_prime_factors) - len(extra_prime_ideal_factors)
    print(f'This gives a {"+" if diff >= 0 else "-"}{diff} advantage.')
    print()
    advantage = len(smooth_ab) - len(siever.prime_siever.p) - len(siever.prime_ideal_siever.p) - 1 + diff
    print(f'The overall advantage, not including characters, is {"+" if advantage >= 0 else "-"}{advantage}.')
    print()
    chars = post_processor.get_quadratic_characters(5, 200)
    print(chars)
    print()
    matrix = post_processor.get_matrix()
    print(matrix)
    solutions = post_processor.solve_matrix()
    print(solutions)
    post_processor.solve_problem(n)
if __name__ == '__main__':
    main()