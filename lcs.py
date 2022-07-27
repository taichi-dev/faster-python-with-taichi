import taichi as ti
import numpy as np

ti.init(arch=ti.cpu)

benchmark = True

N = 8192

f = ti.field(dtype=ti.i32, shape=(N + 1, N + 1))
a = ti.field(dtype=ti.i32, shape=N)
b = ti.field(dtype=ti.i32, shape=N)

@ti.kernel
def initialize(a_input: ti.types.ndarray(), b_input: ti.types.ndarray()):
    for i in range(a_input.shape[0]):
        a[i] = a_input[i]
    for j in range(b_input.shape[0]):
        b[j] = b_input[j]


if benchmark:
    a_numpy = np.random.randint(0, 10, N, dtype=np.int32)
    b_numpy = np.random.randint(0, 10, N, dtype=np.int32)
else:
    a_numpy = np.array([0, 1, 0, 2, 4, 3, 1, 2, 1], dtype=np.int32)
    b_numpy = np.array([4, 0, 1, 4, 5, 3, 1, 2], dtype=np.int32)

initialize(a_numpy, b_numpy)

len_a = a_numpy.shape[0]
len_b = b_numpy.shape[0]

@ti.kernel
def compute_lcs(total_length: ti.i32):
    for i in range(max(1, total_length - len_b), min(total_length, len_a) + 1):
        j = total_length - i
        same = 0
        if a[i - 1] == b[j - 1]:
            same = 1
        f[i, j] = max(f[i - 1, j - 1] + same, max(f[i - 1, j], f[i, j - 1]))

for total_length in range(2, len_a + len_b + 1):
    compute_lcs(total_length)

print(f[len_a, len_b])
