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
        I=100, J=50)
    # num_steps_per_q = 150
    # for q in [29]:
    #     for s in [26]:
    #         (u_1, u_2), (v_1, v_2) = siever.find_basis(q, s)
    #         siever.update_neurons(u_1, u_2, v_1, v_2, False, False)
    #         for i in range(num_steps_per_q):
    #             siever.net.run_steps(1, True)
    #             print(f'Step #{i}')
    #             print(siever.prime_siever.primes_layer.get_last_output())
    #         i =(siever.i_from_spike_time(144))
    #         j =(siever.j_from_spike_time(144))
    #         a = i * u_1 + j * v_1
    #         b = i * u_2 + j * v_2
    #         print(i, j, a, b)
            
    # return
    # siever.net.set_output_neuron(siever.prime_siever.local_bottom_layer)
    results = siever.sieve(prime_upper_bound, ideals_upper_bound, debug=False, verbose=False)
    # for item in results:
    #     for t in item[4]:
    #         print(t)
    print()
    print('raw results')
    print(results)
    print()
    print('filtering...')
    filtered = check_results(siever, results)
    print()
    print('filtered results')
    print(filtered)
if __name__ == '__main__':
    main()