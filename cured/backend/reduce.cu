#include <stdio.h>
#include <cuda_runtime.h>

#include "common.h"
#include "reduce.h"

namespace reduce
{

__global__ void ragged_array_row_wise_sum_reduce_kernel(const int *data, const int ra_size,
    const int *row_starts, const int *row_lengths, const int num_rows, int *out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= num_rows)
  {
    return;
  }

  int row_start = row_starts[gid];
  int row_length = row_lengths[gid];

  for (int i = row_start; i < row_start+row_length; i++)
  {
    out[gid] += data[i];
  }
}

void ragged_array_row_wise_sum_reduce(const int *data, const int ra_size, 
    const int *row_starts, const int *row_lengths, const int num_rows, int *ret)
{
  int *data_d;
  int *row_starts_d;
  int *row_lengths_d;
  int *ret_d;

  cuda_errchk(cudaMalloc(&data_d, sizeof(int)*ra_size));
  cuda_errchk(cudaMalloc(&row_starts_d, sizeof(int)*num_rows));
  cuda_errchk(cudaMalloc(&row_lengths_d, sizeof(int)*num_rows));
  cuda_errchk(cudaMalloc(&ret_d, sizeof(int)*num_rows));

  cuda_errchk(cudaMemcpy(data_d, data, sizeof(int)*ra_size, cudaMemcpyHostToDevice));
  cuda_errchk(cudaMemcpy(row_starts_d, row_starts, sizeof(int)*num_rows, cudaMemcpyHostToDevice));
  cuda_errchk(cudaMemcpy(row_lengths_d, row_lengths, sizeof(int)*num_rows, cudaMemcpyHostToDevice));

  int thread_block_size = 128;
  int num_block = SDIV(num_rows, thread_block_size);
  ragged_array_row_wise_sum_reduce_kernel<<<num_block, thread_block_size>>>(
      data_d, ra_size, row_starts_d, row_lengths_d, num_rows, ret_d);
  cudaDeviceSynchronize();

  cuda_errchk(cudaMemcpy(ret, ret_d, sizeof(int)*num_rows, cudaMemcpyDeviceToHost));

  cuda_errchk(cudaFree(data_d));
  cuda_errchk(cudaFree(row_starts_d));
  cuda_errchk(cudaFree(row_lengths_d));
  cuda_errchk(cudaFree(ret_d));
}

} // namespace reduce
