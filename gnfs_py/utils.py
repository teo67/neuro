import numpy as np
from typing import Any

from sympy import Poly
from sympy.abc import x as var_x
from sympy.polys import polytools
from sympy import GF, FF, ZZ
from collections.abc import Iterable

def check_update_property(new_val: np.ndarray, old_val: np.ndarray, name: str):
    """Do validation on the new value for a property, ensuring that it matches the shape and datatype of the old one.

    Args:
        new_val (np.ndarray): The new value of the property.
        old_val (np.ndarray): The old value of the property.
        name (str): The name of the property, for debugging purposes.
    """
    assert np.shape(new_val) == np.shape(old_val), \
        f'Cannot update property \'{name}\' from shape {np.shape(old_val)} to {np.shape(new_val)}!'
    assert new_val.dtype == old_val.dtype, \
        f'Cannot update property \'{name}\' from type {old_val.dtype} to {new_val.dtype}!'
    
def get_np_array(val: Any, shape: tuple[int], dtype: Any) -> np.ndarray:
    """Given a single value or list/tuple/array and a desired shape & datatype, produce the corresponding array.

    Args:
        val (Any): A single value or list/tuple/array to be converted into an array.
        shape (tuple[int]): The desired shape of the array.
        dtype (Any): The desired datatype of the array.

    Returns:
        np.ndarray: An array with the shape and datatype provided that matches the input value.
    """
    if type(val) == list or type(val) == tuple or type(val) == np.ndarray:
        returning = np.array(val, dtype=dtype)
        assert np.shape(returning) == shape, \
            f'Attempting to make an np array of shape {shape} from an array of shape {np.shape(returning)}!'
        return returning
    return np.full(shape, val, dtype=dtype)

def gcd(a: int, b: int) -> int:
    """Find the greatest common denominator of two integers using the Euclidean algorithm.

    Args:
        a (int): The first integer.
        b (int): The second integer.

    Returns:
        int: The gcd of a and b.
    """
    if a == 0:
        return b
    return gcd(b % a, a)

def primesfrom3to(n: int) -> np.ndarray:
    """Generate an array of all primes within [3, n).

    Args:
        n (int): The exclusive upper bound for generated primes.

    Returns:
        np.ndarray: The output array.
    """
    sieve = np.ones(n//2, dtype=bool)
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i//2]:
            sieve[i*i//2::i] = False
    return 2*np.nonzero(sieve)[0][1::]+1

def primesupto(n: int) -> np.ndarray:
    """Generate an array of all primes up to n.

    Args:
        n (int): The exclusive upper bound for generated primes.

    Returns:
        np.ndarray: The output array.
    """
    return np.concatenate(([2], primesfrom3to(n)), dtype=int)

def pi(x: int) -> int:
    """Find pi(x), or the number of primes up to and including x.

    Args:
        x (int): The inclusive upper bound for counted primes.

    Returns:
        int: The number of primes less than or equal to x.
    """
    return 1 + len(primesfrom3to(x + 1))

def mod_inv(A: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Take the modular inverse of a given array, modulo another array, using the extended Euclidean algorithm.

    Args:
        A (np.ndarray): The array to take the inverse of.
        M (np.ndarray): The modulus.

    Returns:
        np.ndarray: An array of inverses to A (mod M), assuming they exist.
    """
    A, M = np.copy(A), np.copy(M)
    # print(A, M)
    # print("^ above")
    m0 = np.copy(M)
    y = np.zeros(np.shape(A), dtype=int)
    x = np.ones(np.shape(A), dtype=int)
    q = np.zeros(np.shape(A), dtype=int)
    running = np.ones(np.shape(A), dtype=bool)

    while True:
        should_stop = (A <= 1)
        if should_stop.all():
            break
        running[should_stop] = 0
        Ar, Mr = A[running], M[running]
        # print(Ar, Mr)
        q[running] = Ar // Mr
        M[running], A[running] = Ar % Mr, Mr
        x[running], y[running] = y[running], x[running] - q[running] * y[running]
    return np.where(x < 0, x + m0, x)

def legendre(a: int, p: int) -> int:
    """Compute the Legendre symbol (a|p).

    Args:
        a (int): The input integer.
        p (int): The odd prime.

    Returns:
        int: (a|p).
    """
    assert p > 2, f'Can only calculate legendre symbol for p > 2, not p = {p}!'
    return pow(a, (p - 1)//2, p)

def jacobi(n, p):
  assert p > 0 and p % 2 == 1
  n = n % p
  t = 1
  while n != 0:
    while n % 2 == 0:
      n = n // 2
      r = p % 8
      if r == 3 or r == 5:
        t = -t
    n, p = p, n
    if n % 4 == 3 and p % 4 == 3:
      t = -t
    n = n % p
  if p == 1:
    return t
  else:
    return 0
  
# computes a square root of n mod p when p is prime and p > 2
def tonelli(n, p):
    assert p > 2, "p must be greater than 2"
    # https://en.wikipedia.org/wiki/Tonelli%E2%80%93Shanks_algorithm
    # Step 1
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    # We have p - 1 = Q(2^S)
    # Step 2
    z = 0
    while jacobi(z, p) != -1:
        z += 1
        if z >= p:
            raise "Couldn't find any nonresidues??"
    # z is guaranteed to be a nonresidue mod p
    # Step 3
    M = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    R = pow(n, (Q + 1)//2, p)
    # Step 4
    while True:
        if t == 0:
            return 0
        if t == 1:
            return R
        t2i = pow(t, 2, p)
        i = 1
        while t2i != 1 and i < M:
            t2i = pow(t2i, 2, p)
            i += 1
        if t2i != 1:
            return None # n is not a quadratic residue mod p
        b = pow(c, int(pow(2, (M - i - 1))), p)
        M = i
        c = pow(b, 2, p)
        t = (t * c) % p
        R = (R * b) % p

def enumerate_combinations(p: int, d: int):
    if d == 0:
        yield ()
        return
    for combo in enumerate_combinations(p, d - 1):
        for i in range(p):
            yield (i,) + combo
    return

def poly_sqrt(poly: Poly, mod_poly: Poly, primes_with_roots: list[int], debug: bool = False) -> Poly:
    all_primes = primesupto(primes_with_roots[-1] + 1) # we assume primes with roots is ordered, and that we can find an irreducible prime in between
    p_index = 0
    poly_modded = polytools.rem(poly, mod_poly)
    print(f'target: {poly_modded}')
    num_tries = 0
    allowed_tries = 3
    for p in all_primes:
        if primes_with_roots[p_index] > p:
            prime_pow = 1
            poly_to_add = 0
            print(f'Now trying irreducible p = {p}')
            while True:
                prev_pow = prime_pow
                prime_pow *= p

                poly_mod_prime_pow = Poly(poly_modded, modulus=prime_pow)
                if debug:
                    print(f'finding square root of {poly_mod_prime_pow} mod {prime_pow}')
                for combo in enumerate_combinations(p, mod_poly.degree()):
                    p_new = Poly(sum(coeff * var_x**i for i, coeff in enumerate(combo)) * prev_pow + poly_to_add, var_x)
                    square_mod_p = Poly(polytools.rem(p_new * p_new, mod_poly), var_x, modulus=prime_pow)
                    if(square_mod_p == poly_mod_prime_pow):
                        if debug:
                            print(f'found {p_new}')
                        poly_to_add = p_new
                        break
                else:
                    break
                conjugate = sum(prime_pow * var_x ** i for i in range(mod_poly.degree())) - poly_to_add
                candidates = [poly_to_add, conjugate]
                for can in candidates:
                    if debug:
                        print(f'---- trying candidate {can} ----')
                        print(f'square = {polytools.rem(can * can, mod_poly)}, target = {poly_modded}')
                        print('------')
                    if polytools.rem(can * can, mod_poly) == poly_modded:
                        if debug:
                            print(f'Found square root! {can}')
                        return Poly(can, var_x)
                if polytools.trunc(poly_modded, prime_pow) == poly_modded:
                    break
            num_tries += 1
            if num_tries == allowed_tries:
                return None
        else:
            p_index += 1 
    