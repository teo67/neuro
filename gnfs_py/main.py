from sieve.gnf_siever import GNFSiever
from check_results import check_results
import numpy as np
# import utils
# from net.neuron import Neuron, make_neuron
# from net.net import Net
# from net.synapse import Synapse, make_synapse

def main():
    prime_upper_bound = 29 # B_0 in Lattice Sieve paper
    ideals_upper_bound = 103 # B_1 in Lattice Sieve paper
    large_prime_upper_bound = 1000 # B_2 in Lattice Sieve paper
    siever = GNFSiever(
        coefficients=(8, 29, 15, 1), m=31, skew=1,
        b_0_primes=1, b_1_primes=prime_upper_bound, B_primes=np.log(large_prime_upper_bound), # so as to allow up to one large prime
        b_0_ideals=1, b_1_ideals=ideals_upper_bound, B_ideals=np.log(large_prime_upper_bound), B6=10,
        I=20, J=10)
    
    results = siever.sieve(prime_upper_bound, ideals_upper_bound, debug=False, verbose=False)
    print()
    print('raw results')
    print(results)
    print()
    print('filtering...')
    filtered = check_results(siever, results)
    print()
    print('filtered results')
    print(filtered)
    print()
    print(f'Found {len(filtered)} smooth (a, b) pairs.')
if __name__ == '__main__':
    main()