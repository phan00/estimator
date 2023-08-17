#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

void bind_ekf(pybind11::module &m);
