import numpy as np
from typing import Any

def check_update_property(new_val: np.ndarray, old_val: np.ndarray, name: str):
    assert np.shape(new_val) == np.shape(old_val), \
        f'Cannot update property \'{name}\' from shape {np.shape(old_val)} to {np.shape(new_val)}!'
    assert new_val.dtype == old_val.dtype, \
        f'Cannot update property \'{name}\' from type {old_val.dtype} to {new_val.dtype}!'
    
def get_np_array(val: Any, shape: tuple[int], dtype: Any) -> np.ndarray:
    if type(val) == list or type(val) == tuple or type(val) == np.ndarray:
        returning = np.array(val, dtype=dtype)
        assert np.shape(returning) == shape, \
            f'Attempting to make an np array of shape {shape} from an array of shape {np.shape(returning)}!'
        return returning
    return np.full(shape, val, dtype=dtype)

def gcd(a: int, b: int) -> int:
    if a == 0:
        return b
    return gcd(b % a, a)

def primesfrom3to(n: int) -> np.ndarray:
    """ Returns a array of primes, 3 <= p < n """
    sieve = np.ones(n//2, dtype=bool)
    for i in range(3,int(n**0.5)+1,2):
        if sieve[i//2]:
            sieve[i*i//2::i] = False
    return 2*np.nonzero(sieve)[0][1::]+1

def primesupto(n: int) -> np.ndarray:
    return np.concatenate(([2], primesfrom3to(n)))

def pi(x: int) -> int:
    return 1 + len(primesfrom3to(x))

def mod_inv(A: np.ndarray, M: np.ndarray) -> np.ndarray:
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