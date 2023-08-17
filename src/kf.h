#pragma once

#include <armadillo>

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
        M K = Pp*trans(H)*inv(S);
        xc = xp + K * (z - zp);
        Pc = Pp - K * S * trans(K);        
        Pc = (Pc + trans(Pc))/2.;
    }
};

struct KalmanFilter : public KalmanFilterMath {
    arma::Mat<double> State;
    arma::Mat<double> StateCovariance;
    arma::Mat<double> TransitionStateModel;

    arma::Mat<double> ProcessNoise;
    arma::Mat<double> TransitionProcessNoiseModel;

    arma::Mat<double> MeasurementNoise;
    arma::Mat<double> TransitionMeasurementModel;
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

    KalmanFilter(arma::Mat<double> state,
                 arma::Mat<double> covariance,
                 arma::Mat<double> transition_state_model,
                 arma::Mat<double> process_noise,
                 arma::Mat<double> transition_process_noise,
                 arma::Mat<double> transition_measurement_model,
                 arma::Mat<double> measurement_noise) : State(state),
                                                        StateCovariance(covariance),
                                                        TransitionStateModel(transition_state_model),
                                                        TransitionMeasurementModel(transition_measurement_model),
                                                        ProcessNoise(process_noise),
                                                        TransitionProcessNoiseModel(transition_process_noise),
                                                        MeasurementNoise(measurement_noise)
    {
    }

    std::pair<arma::Mat<double>, arma::Mat<double>> predict(const arma::mat& transition_state_model,
                                                            const arma::mat& transition_measurement_model,
                                                            const arma::mat& control_input,
                                                            const arma::mat& control_model) {
        arma::Mat<double> state_predict,
                          covariance_predict;

        KalmanFilterMath::predict<arma::Mat<double>>(this->State,
                                    transition_state_model,
                                    control_model,
                                    control_input,
                                    this->StateCovariance,
                                    this->ProcessNoise,
                                    this->TransitionProcessNoiseModel,
                                    transition_measurement_model,
                                    state_predict,
                                    covariance_predict,
                                    this->MeasurementPredict);
        State = state_predict;
        StateCovariance = covariance_predict;
        return std::make_pair(State, StateCovariance);
    }

    std::pair<arma::Mat<double>, arma::Mat<double>> correct(const arma::mat& transition_measurement_model,
                                                            const arma::mat& measurement) {
        arma::Mat<double> state_correct,
                          covariance_correct;

        KalmanFilterMath::correct<arma::Mat<double>>(this->StateCovariance,
                                                     transition_measurement_model,
                                                     this->MeasurementNoise,
                                                     this->State,
                                                     this->MeasurementPredict,
                                                     measurement,
                                                     state_correct,
                                                     covariance_correct);
        State = state_correct;
        StateCovariance = covariance_correct;
        return std::make_pair(State, StateCovariance);
    }

    std::pair<arma::Mat<double>, arma::Mat<double>> predict(const arma::mat& control_input = arma::mat{},
                                                            const arma::mat& control_model = arma::mat{}) {
        arma::Mat<double> state_predict,
                          covariance_predict,
                          cntrl_input = control_input,
                          cntrl_model = control_model;

        if (cntrl_input.is_empty()) {
            cntrl_input = arma::zeros(this->State.n_rows, this->State.n_cols);
            cntrl_model = arma::zeros(this->TransitionStateModel.n_rows, this->TransitionStateModel.n_cols);
        }

        KalmanFilterMath::predict<arma::Mat<double>>(this->State,
                                    this->TransitionStateModel,
                                    cntrl_model,
                                    cntrl_input,
                                    this->StateCovariance,
                                    this->ProcessNoise,
                                    this->TransitionProcessNoiseModel,
                                    this->TransitionMeasurementModel,
                                    state_predict,
                                    covariance_predict,
                                    this->MeasurementPredict);
        State = state_predict;
        StateCovariance = covariance_predict;
        return std::make_pair(State, StateCovariance);
    }

    std::pair<arma::Mat<double>, arma::Mat<double>> correct(const arma::mat& measurement) {
        arma::Mat<double> state_correct,
                          covariance_correct;

        KalmanFilterMath::correct<arma::Mat<double>>(this->StateCovariance,
                                                     this->TransitionMeasurementModel,
                                                     this->MeasurementNoise,
                                                     this->State,
                                                     this->MeasurementPredict,
                                                     measurement,
                                                     state_correct,
                                                     covariance_correct);
        State = state_correct;
        StateCovariance = covariance_correct;
        return std::make_pair(State, StateCovariance);
    }

};

}


