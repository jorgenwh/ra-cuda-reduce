import numpy as np

import npstructures as nps
import cured

def test_ra_row_wise_sum_reduce():
    for i in range(10):
        ra_shape = np.random.randint(low=1, high=100, size=1000, dtype=np.int32)
        ra_size = np.sum(ra_shape)
        ra_data = np.random.randint(low=-100, high=100, size=ra_size, dtype=np.int32)
        ra = nps.RaggedArray(data=ra_data, shape=ra_shape, dtype=np.int32)
        
        a = np.sum(ra, axis=-1)
        b = cured.reduce.ragged_array_row_wise_sum_reduce(ra)
        np.testing.assert_array_equal(a, b)

