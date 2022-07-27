"""Count the number of primes below a given bound.
"""
use_taichi = True  # set False to use native python

def is_prime(n: int):
    result = True
    for k in range(2, int(n ** 0.5) + 1):
        if n % k == 0:
            result = False
            break
    return result


def count_primes(n: int) -> int:
    count = 0
    for k in range(2, n):
        if is_prime(k):
            count += 1

    return count


if use_taichi:
    import taichi as ti
    ti.init()
    is_prime = ti.func(is_prime)
    count_primes = ti.kernel(count_primes)


import time
info = "use taichi" if use_taichi else "use python"
start = time.perf_counter()
print(count_primes(10000000))
print(f"time elapsed {info}: {time.perf_counter() - start}/s")
