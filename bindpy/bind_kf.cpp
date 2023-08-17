#include "kf.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <carma>

namespace py = pybind11;

class KfBind {
    private:
        Estimator::KalmanFilter kf_;

    public:
        KfBind(py::array_t<double>& state,
               py::array_t<double>& covariance) : kf_{carma::arr_to_mat<double>(state, true),
                                                      carma::arr_to_mat<double>(covariance, true)}
        {
        }

        KfBind(py::array_t<double>& state,
               py::array_t<double>& covariance,
               py::array_t<double>& process_noise,
               py::array_t<double>& transition_process_noise,
               py::array_t<double>& measurement_noise) : kf_{carma::arr_to_mat<double>(state, true),
                                                             carma::arr_to_mat<double>(covariance, true),
                                                             carma::arr_to_mat<double>(process_noise, true),
                                                             carma::arr_to_mat<double>(transition_process_noise, true),
                                                             carma::arr_to_mat<double>(measurement_noise, true)}
        {
        }

        KfBind(py::array_t<double>& state,
               py::array_t<double>& covariance,
               py::array_t<double>& transition_state_model,
               py::array_t<double>& process_noise,
               py::array_t<double>& transition_process_noise,
               py::array_t<double>& transition_measurement_model,
               py::array_t<double>& measurement_noise) : kf_{carma::arr_to_mat<double>(state, true),
                                                             carma::arr_to_mat<double>(covariance, true),
                                                             carma::arr_to_mat<double>(transition_state_model, true),
                                                             carma::arr_to_mat<double>(process_noise, true),
                                                             carma::arr_to_mat<double>(transition_process_noise, true),
                                                             carma::arr_to_mat<double>(transition_measurement_model, true),
                                                             carma::arr_to_mat<double>(measurement_noise, true)}
        {
        }

        py::tuple predict(py::array_t<double>& TransitionStateModel,
                          py::array_t<double>& TransitionMeasurementModel) {
            auto ret = kf_.predict(carma::arr_to_mat<double>(TransitionStateModel),
                                   carma::arr_to_mat<double>(TransitionMeasurementModel));
            return py::make_tuple(
                            carma::mat_to_arr(ret.first),
                            carma::mat_to_arr(ret.second)
                        );
        }

        py::tuple predict(py::array_t<double>& TransitionStateModel,
                          py::array_t<double>& TransitionMeasurementModel,
                          py::array_t<double>& ControlInput,
                          py::array_t<double>& ControlModel) {
            auto ret = kf_.predict(carma::arr_to_mat<double>(TransitionStateModel),
                                   carma::arr_to_mat<double>(TransitionMeasurementModel),
                                   carma::arr_to_mat<double>(ControlInput),
                                   carma::arr_to_mat<double>(ControlModel));
            return py::make_tuple(
                            carma::mat_to_arr(ret.first),
                            carma::mat_to_arr(ret.second)
                        );
        }

        py::tuple correct(py::array_t<double>& TransitionMeasurementModel,
                          py::array_t<double>& Measurement) {
            auto ret = kf_.correct(carma::arr_to_mat<double>(TransitionMeasurementModel),
                                   carma::arr_to_mat<double>(Measurement));
            return py::make_tuple(
                            carma::mat_to_arr(ret.first),
                            carma::mat_to_arr(ret.second)
                        );
        }

        py::tuple predict() {
            auto ret = kf_.predict();
            return py::make_tuple(
                            carma::mat_to_arr(ret.first),
                            carma::mat_to_arr(ret.second)
                        );
        }

        py::tuple correct(py::array_t<double>& Measurement) {
            auto ret = kf_.correct(carma::arr_to_mat<double>(Measurement));
            return py::make_tuple(
                            carma::mat_to_arr(ret.first),
                            carma::mat_to_arr(ret.second)
                        );
        }
};

void bind_kf(py::module &m) {
    py::class_<KfBind>(m, "Kf")
        .def(py::init<py::array_t<double> &, py::array_t<double> &>(), R"pbdoc(
            Initialise Kf.

            Parameters
            ----------
            x0: np.ndarray
               initial state
            P: np.ndarray
               initial covariance
        )pbdoc")
        .def(py::init<py::array_t<double> &,
                      py::array_t<double> &,
                      py::array_t<double> &,
                      py::array_t<double> &,
                      py::array_t<double> &>(), R"pbdoc(
                Initialise Kf.

                Parameters
                ----------
                x: np.ndarray
                   initial state
                P: np.ndarray
                   initial covariance
                Q: np.ndarray
                   process noise
                G: np.ndarray
                   transition process noise
                R: np.ndarray
                   measurement noise
            )pbdoc")
         .def(py::init<py::array_t<double> &,
                        py::array_t<double> &,
                        py::array_t<double> &,
                        py::array_t<double> &,
                        py::array_t<double> &,
                        py::array_t<double> &,
                        py::array_t<double> &>(), R"pbdoc(
                    Initialise Kf.

                    Parameters
                    ----------
                    x: np.ndarray
                       initial state
                    P: np.ndarray
                       initial covariance
                    F: np.ndarray
                       transition state model
                    Q: np.ndarray
                       process noise
                    G: np.ndarray
                       transition process noise
                    H: np.ndarray
                       transition measurement model
                    R: np.ndarray
                       measurement noise
             )pbdoc")
            .def("predict", py::overload_cast<py::array_t<double>&,
                                          py::array_t<double>&>(&KfBind::predict), R"pbdoc(
            Compute predict.

            Parameters
            ----------
            F: np.ndarray
               transition state model
            H: np.ndarray
               transition measurement model
            )pbdoc",
             py::arg("TransitionStateModel"),
             py::arg("TransitionMeasurementModel")
            )
        .def("predict", py::overload_cast<py::array_t<double>&,
                                          py::array_t<double>&,
                                          py::array_t<double>&,
                                          py::array_t<double>&>(&KfBind::predict), R"pbdoc(
                Compute predict.

                Parameters
                ----------
                F: np.ndarray
                    transition state model
                H: np.ndarray
                    transition measurement model
                u: np.ndarray
                    control input
                B: np.ndarray
                    control model
             )pbdoc",
             py::arg("TransitionStateModel"),
             py::arg("TransitionMeasurementModel"),
             py::arg("ControlInput"),
             py::arg("ControlModel")
            )
        .def("predict", py::overload_cast<>(&KfBind::predict), R"pbdoc(
                    Compute predict.
            )pbdoc")
        .def("correct", py::overload_cast<py::array_t<double>&>(&KfBind::correct), R"pbdoc(
            Compute correct.

            Parameters
            ----------
            Measurement: np.ndarray

            )pbdoc",
            py::arg("Measurement")
        );
}
