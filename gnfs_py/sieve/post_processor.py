from sieve.gnf_siever import GNFSiever
import numpy as np
import primefac
import utils
import galois
from sympy import poly, Poly, diff
from sympy.abc import x as var_x
from sympy.polys import polytools

class PostProcessor:
    siever: GNFSiever
    results: list[tuple[int, int, tuple[int, int], tuple[int, int], np.ndarray]]
    smooth_ab: set[tuple[int, int]] | None
    nonsmooth_ab: set[tuple[int, int]] | None
    extra_smooth_ab: set[tuple[int, int]] | None
    extra_prime_factors: set[tuple[int, int]] | None
    extra_prime_ideal_factors: set[tuple[int, int]] | None
    quadratic_characters: set[tuple[int, int]] | None
    matrix: np.ndarray | None
    solutions: list | None
    ordered_prime_factors: list[tuple[int, int]] | None
    ordered_prime_ideal_factors: list[tuple[int, int]] | None
    ordered_quadratic_characters: list[tuple[int, int]] | None
    ordered_ab_pairs: list[tuple[int, int]] | None
    def __init__(self, siever: GNFSiever, results: list[tuple[int, int, tuple[int, int], tuple[int, int], np.ndarray]]):
        """Create a PostProcessor object for validating results.

        Args:
            siever (GNFSiever): The siever that generated the results.
            results (list[tuple[int, int, tuple[int, int], tuple[int, int], np.ndarray]]): The results (see GNFSiever.sieve).
        """
        self.siever = siever
        self.results = results
        self.smooth_ab = None
        self.nonsmooth_ab = None
        self.extra_smooth_ab = None
        self.extra_prime_factors = None
        self.extra_prime_ideal_factors = None
        self.quadratic_characters = None
        self.matrix = None
        self.solutions = None
        self.ordered_prime_factors = None
        self.ordered_prime_ideal_factors = None
        self.ordered_quadratic_characters = None
        self.ordered_ab_pairs = None

    def check_results(self) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
        """Iterate through results from the GNFSiever, validating them and printing information about them.
        Raises an exception if an invalid value was found (i.e. invalid u or v vector, invalid norm, a & b not coprime except q).
        Halts early and returns None if an outlier is found, i.e. an (a, b) pair that has no factors in one of the two prime bases.

        Returns:
            tuple[set[tuple[int, int]], set[tuple[int, int]]] | None: A tuple of the set of smooth (a, b) pairs and the set of non-smooth (a, b) pairs,
            or None if the function halted early.
        """
        self.smooth_ab = set()
        self.nonsmooth_ab = set()
        for q, s, (u_1, u_2), (v_1, v_2), spike_times in self.results:
            for t in spike_times:
                i = self.siever.i_from_spike_time(t)
                j = self.siever.j_from_spike_time(t)
                a = int(i * u_1 + j * v_1)
                b = int(i * u_2 + j * v_2)
                assert (u_1 - s * u_2)%q == 0, f'u = {(u_1, u_2)} was invalid for q {q}, s {s}!'
                assert (v_1 - s * v_2)%q == 0, f'v = {(v_1, v_2)} was invalid for q {q}, s {s}!'
                
                initial_norm = self.siever.prime_ideal_siever.get_norm(a, b)
                assert initial_norm % q == 0, f'norm {initial_norm} was not divisible by special q {q}!'
                ab_gcd = abs(utils.gcd(a, b))
                without_q = False
                if ab_gcd == q: # this is a special case descibed in the Lattice Sieve paper
                    a //= q
                    b //= q
                    ab_gcd = 1
                    without_q = True
                print(f'q:{q}, s:{s}, t:{t} (fired at {t - 2}), i:{i}, j:{j}, a:{a}, b:{b} (u={(u_1, u_2)}, v={(v_1, v_2)}, ab-gcd={ab_gcd})')
                assert ab_gcd == 1, f'{a} and {b} were not coprime! (gcd={ab_gcd})'

                is_fully_smooth = True
                for inner_sieve, name in (self.siever.prime_siever, 'prime'), (self.siever.prime_ideal_siever, 'prime ideal'):
                    if not inner_sieve.check_any_factors(a, b) and not without_q: # not without_q avoids the case where q was in the factor base and was the only prime to factor n
                        print(f'Found an outlier on {name} siever! Norm = {inner_sieve.get_norm(a, b)}')
                        print([p for p in primefac.primefac(int(inner_sieve.get_norm(a, b)))])
                        return None
                    norm = inner_sieve.get_norm(a, b)
                    reduced_norm = inner_sieve.get_reduced_norm(a, b)
                    if inner_sieve == self.siever.prime_ideal_siever and not without_q and reduced_norm % q == 0:
                        reduced_norm //= q
                    is_smooth = abs(reduced_norm) == 1
                    is_fully_smooth = is_fully_smooth and is_smooth
                    print(f'{name} norm {norm} reduced to {reduced_norm}, is smooth = {is_smooth}.')
                if is_fully_smooth:
                    self.smooth_ab.add((a, b))
                else:
                    self.nonsmooth_ab.add((a, b))
        return self.smooth_ab, self.nonsmooth_ab
    
    def find_extra_smooth_pairs(self) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
        """After finding smooth and non-smooth pairs, extract non-smooth pairs which share prime & prime ideal factors with each other to form
        'extra' smooth pairs, composed of both the original factors and some additional factors.

        Returns:
            tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None: A tuple of extra (a, b) pairs, the set of (r, p) prime factors
            used to make them (excluding the original prime factor base), and the set of (r, p) prime ideal factors used to make them (excluding the original prime ideal factor base).
            Returns None if the function halted early because it was called before generating smooth/non-smooth pairs.
        """
        if self.smooth_ab is None or self.nonsmooth_ab is None:
            print('Cannot find extra smooth pairs without finding smooth/non-smooth pairs first! See PostProcessors.check_results.')
            return None
        prime_factor_counts: dict[tuple[int, int], set[tuple[int, int]]] = {}
        prime_ideal_factor_counts: dict[tuple[int, int], set[tuple[int, int]]] = {}
        extra_smooth: set[tuple[int, int]] = self.nonsmooth_ab.copy()
        for a, b in self.nonsmooth_ab:
            for inner_siever, count in (self.siever.prime_siever, prime_factor_counts), (self.siever.prime_ideal_siever, prime_ideal_factor_counts):
                reduced_norm = round(abs(inner_siever.get_reduced_norm(a, b)))

                if inner_siever == self.siever.prime_ideal_siever:
                    roots_and_factors = []
                    for factor in primefac.primefac(reduced_norm):
                        if factor == 1:
                            continue
                        roots = inner_siever.find_roots(factor)
                        for root in roots:
                            if (a - b * root) % factor == 0:
                                roots_and_factors.append((root, factor))
                    factors = roots_and_factors
                else:
                    factors = [(self.siever.prime_siever.m, p) for p in primefac.primefac(reduced_norm) if p != 1]
                
                for r, p in factors:
                    if (r, p) not in count:
                        count[(r, p)] = set()
                    count[(r, p)].add((a, b))
                
                if len(factors) == 0:
                    extra_smooth.discard((a, b))
        self.__reduce_extra_smooth_candidates(extra_smooth, prime_factor_counts, prime_ideal_factor_counts)
        self.extra_smooth_ab = extra_smooth
        self.extra_prime_factors = set(factor for factor in prime_factor_counts if len(prime_factor_counts[factor]) > 0)
        self.extra_prime_ideal_factors = set(factor for factor in prime_ideal_factor_counts if len(prime_ideal_factor_counts[factor]) > 0)
        return self.extra_smooth_ab, self.extra_prime_factors, self.extra_prime_ideal_factors

    def __reduce_extra_smooth_candidates(self, candidates: set[tuple[int, int]], 
                                         prime_factor_counts: dict[tuple[int, int], set[tuple[int, int]]], 
                                         prime_ideal_factor_counts: dict[tuple[int, int], set[tuple[int, int]]]):
        """(private method) A helper function to reduce the set of (a, b) candidates for 'extra' smooth pairs into only those that share both prime and prime ideal factors.

        Args:
            candidates (set[tuple[int, int]]): The incoming set of candidates for 'extra' smooth pairs.
            prime_factor_counts (dict[tuple[int, int], set[tuple[int, int]]]): A dictionary mapping (root, prime) pairs outside of the prime factor base to the candidates that they factor.
            prime_ideal_factor_counts (dict[tuple[int, int], set[tuple[int, int]]]): A dictionary mapping (root, prime ideal) pairs outside of the prime ideal factor base to the candidates that they factor.
        """
        removed_any = False
        for counts in prime_factor_counts, prime_ideal_factor_counts:
            for r, p in counts:
                cans = counts[(r, p)] & candidates
                if len(cans) == 1:
                    candidates.remove(cans.pop())
                    removed_any = True
                counts[(r, p)] = cans
        if removed_any:
            self.__reduce_extra_smooth_candidates(candidates, prime_factor_counts, prime_ideal_factor_counts)

    def validate_extra_smooth_pairs(self) -> bool:
        """Check that all 'extra' smooth pairs are in fact smooth in the context of the extra factors generated on the side and print them out. This should return True unless there is a bug.

        Returns:
            bool: True if all extra pairs were smooth, False otherwise. Returns False if the function halted early because extra pairs had not yet been generated.
        """
        if self.extra_smooth_ab is None or self.extra_prime_factors is None or self.extra_prime_ideal_factors is None:
            print('Cannot validate extra smooth pairs if they have not been generated! See PostProcessor.find_extra_smooth_pairs.')
            return False
        for (a, b) in self.extra_smooth_ab:
            str_ = f'Validating extra smooth pair ({a}, {b})...'
            for inner_siever, extra_factors, name in (self.siever.prime_siever, self.extra_prime_factors, 'prime'), (self.siever.prime_ideal_siever, self.extra_prime_ideal_factors, 'prime ideal'):
                initial_norm = inner_siever.get_reduced_norm(a, b)
                reduced_norm = initial_norm
                for (r, p) in extra_factors:
                    if (a - r * b)%p == 0:
                        while reduced_norm%p == 0:
                            reduced_norm //= p
                str_ += f' [{name} norm: {initial_norm} -> {reduced_norm}] '
                if abs(reduced_norm) != 1:
                    print(str_)
                    print('Found an outlier!')
                    return False
            print(str_)
        return True

    def get_quadratic_characters(self, num_characters: int, upper_prime_bound: int) -> set[tuple[int, int]] | None:
        if self.extra_smooth_ab is None or self.smooth_ab is None:
            print('Cannot generate the quadratic characters without smooth and extra smooth factors! See PostProcessor.check_results.')
            return None
        characters = set()
        all_primes = utils.primesupto(upper_prime_bound)
        for p in all_primes:
            if p <= self.siever.prime_ideal_siever.b_1:
                continue # these will all already be in the factor base
            roots = self.siever.prime_ideal_siever.find_roots(p)
            for root in roots:
                if (root, p) not in self.extra_prime_ideal_factors:
                    characters.add((root, int(p)))
                    if len(characters) == num_characters:
                        self.quadratic_characters = characters
                        return characters
        assert False, f'Upper prime bound of {upper_prime_bound} was not large enough to generate {num_characters} quadratic characters!'
    
    def get_matrix(self) -> galois.FieldArray | None:
        if self.extra_smooth_ab is None or self.smooth_ab is None or self.quadratic_characters is None:
            print('Cannot generate the matrix without smooth factors, extra smooth factors, and quadratic characters! See PostProcessor.check_results.')
            return None
        self.ordered_prime_factors = list(set(zip(self.siever.prime_siever.r, self.siever.prime_siever.p)) | self.extra_prime_factors)
        self.ordered_prime_ideal_factors = list(set(zip(self.siever.prime_ideal_siever.r, self.siever.prime_ideal_siever.p)) | self.extra_prime_ideal_factors)
        self.ordered_quadratic_characters = list(self.quadratic_characters)
        self.ordered_ab_pairs = list(self.smooth_ab | self.extra_smooth_ab)
        mat_height = len(self.ordered_prime_factors) + len(self.ordered_prime_ideal_factors) + len(self.ordered_quadratic_characters) + 1
        mat_width = len(self.ordered_ab_pairs)
        assert mat_width > mat_height, f'There is no advantage (width = {mat_width} <= {mat_height} = height) so we cannot guarantee a solution!'
        mat = np.zeros((mat_height, mat_width), dtype=int)
        for i, (a, b) in enumerate(self.ordered_ab_pairs):
            # each will correspond to a column
            for inner_siever, factors, offset in (self.siever.prime_siever, self.ordered_prime_factors, 0), (self.siever.prime_ideal_siever, self.ordered_prime_ideal_factors, len(self.ordered_prime_factors)):
                norm = inner_siever.get_norm(a, b)
                for j, (r, p) in enumerate(factors):
                    if (a - r * b)%p == 0:
                        num_divisions = 0
                        while norm%p == 0:
                            norm //= p
                            num_divisions += 1
                        mat[j + offset][i] = num_divisions
            for j, (s, q) in enumerate(self.ordered_quadratic_characters):
                legend = utils.legendre(a + b * s, q)
                if legend != 1:
                    mat[j + len(self.ordered_prime_factors) + len(self.ordered_prime_ideal_factors)][i] = 1
        self.matrix = mat
        return mat
    
    def solve_matrix(self) -> galois.FieldArray | None:
        if self.matrix is None:
            print('No matrix has been generated! See PostProcessor.get_matrix.')
            return None
        gf = galois.GF(2)
        gf_mat = gf(self.matrix%2)
        self.solutions = sorted(gf_mat.null_space(), key=lambda row: len(np.where(row)[0]))
        return self.solutions
    
    def solve_problem(self, n: int):
        if self.solutions is None:
            print('Cannot solve the problem before solving the matrix! See PostProcessor.solve_matrix.')
            return None
        mod_poly = self.siever.prime_ideal_siever.sympy_poly
        for solution in self.solutions:
            print(solution)
            hits = self.matrix @ np.array(solution) // 2
            building_poly = 1
            for i in range(len(self.ordered_prime_ideal_factors)):
                for _ in hits[i]:
                    building_poly = polytools.rem(building_poly * (a - b * var_x), mod_poly, modulus=n)
                    building_poly = polytools.trunc(building_poly, n)
            break
            # building_poly = 1
            # indices = np.where(solution)[0]
            # print(indices)
            # y = 1
            # for i in indices:
            #     a, b = self.ordered_ab_pairs[i]
            #     building_poly = polytools.rem(building_poly * (a - b * var_x), mod_poly, modulus=n)
            #     building_poly = polytools.trunc(building_poly, n)
            #     y *= (a - b * self.siever.prime_siever.m)
            #     y %= n
            # y *= (diff(mod_poly).subs({var_x: self.siever.prime_siever.m}))
            # y %= n
            # print(building_poly)
            # x = building_poly.subs({var_x: self.siever.prime_siever.m}) % n
            # print(x, y)
            # print((x**2)%n, (y**2)%n)

