import time
import numpy as np
import cupy as cp

import npstructures as nps
from cured.reduce import ragged_array_row_wise_sum_reduce

use_cp = 0

xp = np
if use_cp:
    xp = cp
    nps.set_backend(cp)

ra_shape = np.random.randint(low=1, high=20, size=1000000)
ra_size = np.sum(ra_shape)
ra_data = np.random.randint(low=-50, high=50, size=ra_size, dtype=np.int32)
print("Data is initialized")

if use_cp:
    ra_shape = cp.asanyarray(ra_shape)
    ra_data = cp.asanyarray(ra_data)
    print("Data moved to device")

ra = nps.RaggedArray(data=ra_data, shape=ra_shape, dtype=np.int32)
print("Ragged array is ready")

print("Performing row-wise sum")
t1 = time.perf_counter()
if use_cp:
    print(f"Using CUDA solution for {'cupy' if use_cp else 'numpy'} RA")
    sums = ragged_array_row_wise_sum_reduce(ra)
else:
    print(f"Using np.sum solution for {'cupy' if use_cp else 'numpy'} RA")
    sums = np.sum(ra, axis=-1)
t2 = time.perf_counter()
elapsed = t2 - t1
print(f"Time spent summing ragged array rows with {'cupy' if use_cp else 'numpy'} backend module: {elapsed}")
