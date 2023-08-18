#include "ekf.h"

#include <gtest/gtest.h>

TEST (EKF, ExtendedKalmanFilterMath) {
    // init
    auto stateModel = [](const arma::mat& s, double t) {
        arma::mat F = {{1, t, 0, 0},
                       {0, 1, 0, 0},
                       {0, 0, 1, t},
                       {0, 0, 0, 1}};
        arma::mat sn = F*s;
        return sn;
    };

    auto measureModel = [](const arma::mat& s) {
        double angle = atan2(s(2), s(0));
        double range = sqrt(s(0)*s(0) + s(2)*s(2));
        arma::mat r = trans(arma::mat{angle, range});
        return r;
    };

    double t = 0.2;
    arma::mat S  = Utils::cholPSD(diagmat(arma::mat{100, 1e3, 100, 1e3}));
    arma::mat x  = trans(arma::mat{35., 0., 45., 0.});
    arma::mat Qs = Utils::cholPSD(diagmat(arma::mat{0, .01, 0, .01}));
    arma::mat z  = arma::colvec{0.926815, 50.2618};
    arma::mat Rs = Utils::cholPSD(diagmat(arma::mat{2e-6, 1.}));

    using namespace Estimator;
    ExtendedKalmanFilterMath f;

    // predict
    auto pred = f.predict(Qs, x, S, stateModel, nullptr, t);
    {
        arma::colvec expectedPredState {35., 0., 45., 0.};
        arma::mat expectedPredSqrtCov  {{-11.8322,       0.,       0.,       0.},
                                        {-16.9031, -26.7263,       0.,       0.},
                                        {      0.,       0., -11.8322,       0.},
                                        {      0.,       0., -16.9031, -26.7263}};
        arma::mat expectedPredJacobian {{1.,0.2, 0., 0.},
                                        {0., 1., 0., 0.},
                                        {0., 0., 1.,0.2},
                                        {0., 0., 0., 1.}};

        ASSERT_TRUE(approx_equal(pred.x, expectedPredState, "reldiff", 0.00001));
        ASSERT_TRUE(approx_equal(pred.S, expectedPredSqrtCov, "reldiff", 0.00001));
        ASSERT_TRUE(approx_equal(pred.dFdx, expectedPredJacobian, "reldiff", 0.00001));
    }

    // correct
    auto corr = f.correct(z, Rs, pred.x, pred.S, measureModel, nullptr);
    {
        arma::colvec expectedCorrState { 30.1194, -6.97231, 40.3092, -6.70121 };
        arma::mat  expectedCorrSqrtCov {{0.615061,            0,            0,            0},
                                        {0.878658,      26.7263,            0,            0},
                                        {0.777205, -6.04034e-20,     0.130612,            0},
                                        { 1.11029, -8.62905e-20,     0.186588,      26.7263}};

        ASSERT_TRUE(approx_equal(corr.first, expectedCorrState, "reldiff", 0.00001));
        ASSERT_TRUE(approx_equal(corr.second, expectedCorrSqrtCov, "absdiff", 0.0001));
    }

}
