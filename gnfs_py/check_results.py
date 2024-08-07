from sieve.gnf_siever import GNFSiever
import numpy as np
import utils
import primefac

def check_results(siever: GNFSiever, results: list[tuple[int, int, tuple[int, int], tuple[int, int], np.ndarray]]) -> set[tuple[int, int]]:
    smooth_ab = set()
    for q, s, (u_1, u_2), (v_1, v_2), spike_times in results:
        for t in spike_times:
            i = siever.i_from_spike_time(t)
            j = siever.j_from_spike_time(t)
            a = i * u_1 + j * v_1
            b = i * u_2 + j * v_2
            assert (u_1 - s * u_2)%q == 0, f'u = {(u_1, u_2)} was invalid for q {q}, s {s}!'
            assert (v_1 - s * v_2)%q == 0, f'v = {(v_1, v_2)} was invalid for q {q}, s {s}!'
            
            initial_norm = siever.prime_ideal_siever.get_norm(a, b)
            assert initial_norm % q == 0, f'norm {initial_norm} was not divisible by special q {q}!'
            ab_gcd = abs(utils.gcd(a, b))
            without_q = False
            if ab_gcd == q:
                a //= q
                b //= q
                ab_gcd = 1
                without_q = True
            print(f'q:{q}, s:{s}, t:{t} (fired at {t - 2}), i:{i}, j:{j}, a:{a}, b:{b} (u={(u_1, u_2)}, v={(v_1, v_2)}, ab-gcd={ab_gcd})')
            assert ab_gcd == 1, f'{a} and {b} were not coprime! (gcd={ab_gcd})'

            is_fully_smooth = True
            for inner_sieve, name in (siever.prime_siever, 'prime'), (siever.prime_ideal_siever, 'prime ideal'):
                if not inner_sieve.check_any_factors(a, b) and not without_q: # not without_q avoids the case where q was in the factor base and was the only prime to factor n
                    print(f'Found an outlier on {name} siever! Norm = {inner_sieve.get_norm(a, b)}')
                    print([p for p in primefac.primefac(int(inner_sieve.get_norm(a, b)))])
                    return None
                norm = inner_sieve.get_norm(a, b)
                reduced_norm = inner_sieve.get_reduced_norm(a, b)
                if inner_sieve == siever.prime_ideal_siever and not without_q and reduced_norm % q == 0:
                    reduced_norm //= q
                is_smooth = abs(reduced_norm) == 1
                is_fully_smooth = is_fully_smooth and is_smooth
                print(f'{name} norm {norm} reduced to {reduced_norm}, is smooth = {is_smooth}.')
            if is_fully_smooth:
                smooth_ab.add((a, b))
    return smooth_ab