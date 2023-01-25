import numpy as np
import cupy as cp

from cured_backend import ra_row_wise_sum_reduce
from cured_backend import cp_ra_row_wise_sum_reduce

def ragged_array_row_wise_sum_reduce(ra):
    data = ra.ravel()
    row_starts = ra._shape.starts
    row_lengths = ra._shape.lengths

    cuda_input = isinstance(data, cp.ndarray)

    if cuda_input:
        ret = cp.zeros(len(row_starts), dtype=np.int32)
        cp_ra_row_wise_sum_reduce(data.data.ptr, data.size, 
                row_starts.data.ptr, row_lengths.data.ptr, row_starts.size, ret.data.ptr)
        return ret

    return ra_row_wise_sum_reduce(data, row_starts, row_lengths)
