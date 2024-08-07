import numpy as np
from typing import Any

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