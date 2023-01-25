#ifndef REDUCE_H_
#define REDUCE_H_

#include <cuda_runtime.h>

#include "common.h"

namespace reduce
{

void ragged_array_row_wise_sum_reduce(const int *data, const int ra_size, 
    const int *row_starts_data, const int *row_lengths_data, const int num_rows, int *ret_data, const bool on_device);

} // namespace reduce

#endif // REDUCE_H_
