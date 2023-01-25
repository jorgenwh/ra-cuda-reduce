#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "reduce.h"

namespace py = pybind11;

PYBIND11_MODULE(cured_backend, m) 
{
  m.doc() = "Documentation for the cured backend module";

  m.def("ra_row_wise_sum_reduce", [](const py::array_t<int> &ra_data, 
        const py::array_t<int> &row_starts, const py::array_t<int> &row_lengths) 
  {
    const int *data = ra_data.data();
    const int ra_size = ra_data.size();
    const int *row_starts_data = row_starts.data();
    const int *row_lengths_data = row_lengths.data();
    const int num_rows = row_starts.size();
    
    auto ret = py::array_t<int>(num_rows);
    int *ret_data = ret.mutable_data();

    reduce::ragged_array_row_wise_sum_reduce(
        data, ra_size, row_starts_data, row_lengths_data, num_rows, ret_data);

    return ret;
  });
  m.def("cp_ra_row_wise_sum_reduce", []()
  {
    return 0;
  });
}
