#include "ekf.h"
#include "models.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <carma>

namespace py = pybind11;

class EkfBind {
    private:
        Estimator::ExtendedKalmanFilter ekf_;

    public:
        EkfBind(py::array_t<double>& state,
                py::array_t<double>& covariance,
                py::array_t<double>& processNoise,
                py::array_t<double>& measureNoise) : ekf_{carma::arr_to_mat<double>(state, true),
                                                          carma::arr_to_mat<double>(covariance, true),
                                                          carma::arr_to_mat<double>(processNoise, true),
                                                          carma::arr_to_mat<double>(measureNoise, true)}
        {
        }

        py::tuple predictStateModel(double dt) {
            auto ret = ekf_.predict(Models::stateModel<arma::Mat<double>>, dt);
            return py::make_tuple(carma::mat_to_arr(ret.first),
                                  carma::mat_to_arr(ret.second));
        }

        py::tuple correctMeasureModel(py::array_t<double>& measurement) {
            arma::Mat<double> z = carma::arr_to_mat(measurement);
            auto ret = ekf_.correct(z, Models::measureModel<arma::Mat<double>>, z);
            return py::make_tuple(carma::mat_to_arr(ret.first),
                                  carma::mat_to_arr(ret.second));
        }
};

void bind_ekf(py::module &m) {
    py::class_<EkfBind>(m, "Ekf")
        .def(py::init<py::array_t<double> &,
                      py::array_t<double> &,
                      py::array_t<double> &,
                      py::array_t<double> &>(), R"pbdoc(
                Initialize Ekf.

                Parameters
                ----------
                x: np.ndarray
                   initial state
                P: np.ndarray
                   initial covariance
                Q: np.ndarray
                   process noise
                R: np.ndarray
                   measurement noise
            )pbdoc")
        .def("predictStateModel", py::overload_cast<double>(&EkfBind::predictStateModel), R"pbdoc(
                Compute predict.

                Parameters
                ----------
                delta_time: double
                    time step
             )pbdoc",
             py::arg("delta_time")
            )
        .def("correctMeasureModel", py::overload_cast<py::array_t<double>&>(&EkfBind::correctMeasureModel), R"pbdoc(
                Compute correct.

                Parameters
                ----------
                Measurement: np.ndarray
             )pbdoc",
             py::arg("measurement")
            );
}
