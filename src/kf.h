#pragma once

#include <armadillo>
#include <carma>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>


namespace Estimator {

struct KalmanFilterMath
{
    template <class M = arma::mat>
    void predict(const M& x,
                 const M& A,
                 const M& B,
                 const M& u,
                 const M& P,
                 const M& Q,
                 const M& G,
                 const M& H,
                 M& xp,
                 M& Pp,
                 M& zp) {

        G.print("G");
        Q.print("Q");
        xp = A*x + B*u;
        Pp = A*P*trans(A) + G*Q*trans(G);
        Pp = (Pp + trans(Pp))/2.;
        zp = H*x;
    }

    template <class M = arma::mat>
    void correct(const M& Pp,
                 const M& H,
                 const M& R,
                 const M& xp,
                 const M& zp,
                 const M& z,
                 M& xc,
                 M& Pc) {
        M S = H*Pp*trans(H) + R;
        M K = Pp*H*inv(S);
        xc = xp + K * (z - zp);
        Pc = Pp - K * S * trans(K);
        Pc = (Pc + trans(Pc))/2.;
    }
};

struct KalmanFilter : public KalmanFilterMath {
    arma::Mat<double> State;
    arma::Mat<double> StateCovariance;

    arma::Mat<double> ProcessNoise;
    arma::Mat<double> TransitionProcessNoiseModel;

    arma::Mat<double> MeasurementNoise;
    arma::Mat<double> MeasurementPredict;

    KalmanFilter(arma::Mat<double> state,
                 arma::Mat<double> covariance) : State(state),
                                                 StateCovariance(covariance),
                                                 ProcessNoise(arma::mat{0.0}),
                                                 TransitionProcessNoiseModel(arma::mat{0.0}),
                                                 MeasurementNoise(arma::mat{1.0})
    {
    }

    KalmanFilter(arma::Mat<double> state,
                 arma::Mat<double> covariance,
                 arma::Mat<double> process_noise,
                 arma::Mat<double> transition_process_noise,
                 arma::Mat<double> measurement_noise) : State(state),
                                                        StateCovariance(covariance),
                                                        ProcessNoise(process_noise),
                                                        TransitionProcessNoiseModel(transition_process_noise),
                                                        MeasurementNoise(measurement_noise)
    {
    }

    std::pair<arma::Mat<double>, arma::Mat<double>> predict(const arma::mat& TransitionStateModel,
                                                            const arma::mat& TransitionMeasurementModel,
                                                            const arma::mat& ControlInput = arma::mat{0.0},
                                                            const arma::mat& ControlModel = arma::mat{0.0}) {
        arma::Mat<double> StatePredict,
                          CovariancePredict;

        KalmanFilterMath::predict<arma::Mat<double>>(State,
                                    TransitionStateModel,
                                    ControlModel,
                                    ControlInput,
                                    StateCovariance,
                                    ProcessNoise,
                                    TransitionProcessNoiseModel,
                                    TransitionMeasurementModel,
                                    StatePredict,
                                    CovariancePredict,
                                    MeasurementPredict);
        State = StatePredict;
        StateCovariance = CovariancePredict;
        return std::make_pair(State, StateCovariance);
    }

    std::pair<arma::Mat<double>, arma::Mat<double>> correct(const arma::mat& TransitionMeasurementModel,
                                                            const arma::mat& Measurement) {
        arma::Mat<double> StateCorrect,
                          CovarianceCorrect;

        KalmanFilterMath::correct<arma::Mat<double>>(StateCovariance,
                                                     TransitionMeasurementModel,
                                                     MeasurementNoise,
                                                     State,
                                                     MeasurementPredict,
                                                     Measurement,
                                                     StateCorrect,
                                                     CovarianceCorrect);
        State = StateCorrect;
        StateCovariance = CovarianceCorrect;
        return std::make_pair(State, StateCovariance);
    }

};

}


