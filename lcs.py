import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

benchmark = True

N = 15000

f = ti.field(dtype=ti.i32, shape=(N + 1, N + 1))

if benchmark:
    a_numpy = np.random.randint(0, 10, N, dtype=np.int32)
    b_numpy = np.random.randint(0, 10, N, dtype=np.int32)
else:
    a_numpy = np.array([0, 1, 0, 2, 4, 3, 1, 2, 1], dtype=np.int32)
    b_numpy = np.array([4, 0, 1, 4, 5, 3, 1, 2], dtype=np.int32)


@ti.kernel
def compute_lcs(a: ti.types.ndarray(), b: ti.types.ndarray()) -> ti.i32:
    len_a, len_b = a.shape[0], b.shape[0]

    ti.loop_config(serialize=True)
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            same = 0
            if a[i - 1] == b[j - 1]:
                same = 1
            f[i, j] = max(f[i - 1, j - 1] + same, max(f[i - 1, j], f[i,
                                                                     j - 1]))

    return f[len_a, len_b]


print(compute_lcs(a_numpy, b_numpy))
