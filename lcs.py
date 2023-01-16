import taichi as ti
import numpy as np
import time

ti.init(arch=ti.cpu)

benchmark = False

N = 15000

f = ti.field(dtype=ti.i32, shape=(N + 1, N + 1))

if benchmark:
    a_numpy = np.random.randint(0, 100, N, dtype=np.int32)
    b_numpy = np.random.randint(0, 100, N, dtype=np.int32)
    f_numpy = np.zeros((N + 1, N + 1), dtype=np.int32)
else:
    a_numpy = np.array([0, 1, 0, 2, 4, 3, 1, 2, 1], dtype=np.int32)
    b_numpy = np.array([4, 0, 1, 4, 5, 3, 1, 2], dtype=np.int32)


@ti.kernel
def compute_lcs(a: ti.types.ndarray(), b: ti.types.ndarray()) -> ti.i32:
    len_a, len_b = a.shape[0], b.shape[0]

    ti.loop_config(serialize=True)
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            f[i, j] = ti.max(f[i - 1, j - 1] + (a[i - 1] == b[j - 1]),
                          ti.max(f[i - 1, j], f[i, j - 1]))

    return f[len_a, len_b]


def compute_lcs_numpy(a, b) -> int:
    len_a, len_b = a.shape[0], b.shape[0]

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            f_numpy[i, j] = max(f_numpy[i - 1, j - 1] + (a[i - 1] == b[j - 1]),
                                max(f_numpy[i - 1, j], f_numpy[i, j - 1]))

    return f_numpy[len_a, len_b]


if benchmark:
    t0 = time.perf_counter()
    print(compute_lcs(a_numpy, b_numpy))
    t1 = time.perf_counter()
    print(f"Time cost using Taichi: {t1 - t0}s")
    print(compute_lcs_numpy(a_numpy, b_numpy))
    t2 = time.perf_counter()
    print(f"Time cost using NumPy: {t2 - t1}s")
else:
    print(compute_lcs(a_numpy, b_numpy))
