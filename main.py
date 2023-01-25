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

ra_shape = xp.random.randint(low=1, high=1000, size=1000000)
ra_size = xp.sum(ra_shape)
ra_data = xp.random.randint(low=-50, high=50, size=ra_size, dtype=np.int32)

ra = nps.RaggedArray(data=ra_data, shape=ra_shape, dtype=np.int32)

t1 = time.perf_counter()
if use_cp:
    sums = ragged_array_row_wise_sum_reduce(ra)
else:
    sums = np.sum(ra, axis=-1)
t2 = time.perf_counter()
elapsed = t2 - t1
print(f"time spent summing ragged array rows with {'cupy' if use_cp else 'numpy'} backend module: {elapsed}")
