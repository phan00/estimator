#include "models.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <carma>

void bind_models(py::module &m) {
    m.def("stateModel", [](py::array_t<double> x, double dt){
        arma::Mat<double> xm = carma::arr_to_mat(x);
        arma::Mat<double> r  = Models::stateModel<arma::Mat<double>>(xm, dt);
        return carma::mat_to_arr(r);
    },R"pbdoc(
          stateModel
      )pbdoc",
      py::arg("x"),
      py::arg("dt")
    );
    m.def("measureModel", [](py::array_t<double> x){
          arma::Mat<double> xm = carma::arr_to_mat(x);
          arma::Mat<double> r  = Models::measureModel<arma::Mat<double>>(xm);
          return carma::mat_to_arr(r);
    },R"pbdoc(
            measureModel
        )pbdoc",
        py::arg("x")
    );
}
