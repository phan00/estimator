#pragma once

#include <armadillo>

namespace Utils {

template <typename Type>
arma::Col<Type> diag (const arma::Mat<Type>& x) {
    return x.diag();
}

inline double eps (const double& x = 1.0) {
    auto xp = std::abs(x);
    return std::nextafter(xp, xp + 1.0f) - xp;
}

inline bool isSymmetricPositiveSemiDefinite (const std::string& name, const arma::Mat<double>& x) {

    if (x.is_empty()) {
        throw std::invalid_argument("Matrix " + name + " is empty.");
    }

    if (!x.is_square()) {
        throw std::invalid_argument("Matrix " + name + " isn't square.");
    }

    auto tol = 100.*max(eps(abs(diag(x))));

    if (!x.is_symmetric()) {
        arma::mat d = abs(x - trans(x));
        if (!find(d>sqrt(tol)).is_empty()) {
            std::stringstream ss;
            ss << "sqrt(tol)=" << sqrt(tol) << std::endl;
            ss << "abs(x - trans(x))=" << std::endl << d << std::endl;
            ss << "Matrix: " << std::endl << x << std::endl << " isn't symmetric.";
            throw std::invalid_argument(ss.str());
        }
    }

    auto notPositiveSemidefinite = any(find(eig_sym((x + x.t())/2.) < -tol ));

    if (notPositiveSemidefinite) {
        throw std::invalid_argument("Matrix " + name + " isn't positive definite.");
    }

    return true;
}

inline arma::mat svdPSD(const arma::mat &A) {
    arma::mat U,V;
    arma::vec s;
    svd(U,s,V,A);
    return V*sqrt(diagmat(s));
}

inline arma::Mat<double>
qrFactor(const arma::Mat<double>& A,
         const arma::Mat<double>& S,
         const arma::Mat<double>& Ns) {
    arma::Mat<double> D = join_cols(S.t()*A.t(),Ns.t()), Q, R;
    qr_econ(Q, R, D);
    return R.t();
}

inline arma::Mat<double>
cholPSD(const arma::Mat<double>& A) {
    arma::Mat<double> ret;
    if (!chol(ret, A)) {
        return svdPSD(A);
    }
    return ret.t();
}

inline size_t length(const arma::Mat<double>& m) {
    return m.n_rows * m.n_cols;
}

inline arma::Mat<double>
zeros(size_t n, size_t m) {
    return arma::zeros(n, m);
}

inline arma::Mat<double> ComputeKalmanGain(arma::Mat<double> Sy,
                                           arma::Mat<double> Pxy) {
    arma::Mat<double> K1 = arma::solve(arma::trimatl(Sy), trans(Pxy), arma::solve_opts::fast);
    arma::Mat<double> K  = arma::trans(arma::solve(arma::trimatu(trans(Sy)), K1, arma::solve_opts::fast));
    return K;
}
/*
arma::mat global2localcoordjac(const arma::vec& tgtpos_,
                               const arma::vec& sensorpos,
                               const arma::mat& laxes) {
    arma::vec tgtpos = tgtpos_.rows(0,Size(sensorpos,1)-1);
    auto relpos = tgtpos - sensorpos;
    arma::vec relposlocal = trans(laxes) * relpos;

    size_t pos_x = 0,
           pos_y = 1,
           pos_z = 2;
    size_t pos_a = 0,
           pos_e = 1,
           pos_r = 2;

    auto xrel = relposlocal (pos_x);
    auto yrel = relposlocal (pos_y);
    auto zrel = relposlocal (pos_z);

    auto xysq =std::pow(xrel, 2.0) + std::pow(yrel, 2.0);
    auto xyzsq=xysq + std::pow(zrel, 2.0);
    auto A = zeros(Size(laxes));

    if (xyzsq == 0) {
        A (pos_r, pos_x) = 1.0;
        A (pos_r, pos_y) = 1.0;
        A (pos_r, pos_z) = 1.0;
        A = laxes * A;
    } else if (xysq == 0) {
        auto x = toDeg( -1.0 / zrel );
        A (pos_e, pos_x) = x;
        A (pos_e, pos_y) = x;
        A (pos_r, pos_z) = 1.0;
        A = laxes * A;
    } else {
        A(pos_a, pos_x) = -yrel/xysq;
        A(pos_a, pos_y) = xrel/xysq;

        A(pos_e,pos_x)  = -xrel*zrel/sqrt(xysq)/xyzsq;
        A(pos_e,pos_y)  = -yrel*zrel/sqrt(xysq)/xyzsq;
        A(pos_e,pos_z)  = xysq/sqrt(xysq)/xyzsq;

        A(pos_r,pos_x)  = xrel/sqrt(xyzsq);
        A(pos_r,pos_y)  = yrel/sqrt(xyzsq);
        A(pos_r,pos_z)  = zrel/sqrt(xyzsq);

        for (size_t r=0; r<A.n_rows; ++r) {
            A.row(r) = trans(laxes * arma::mat(trans(A.row(r))));
        }
        A.row(pos_a) = toDeg(arma::mat(A.row(pos_a)));
        A.row(pos_e) = toDeg(arma::mat(A.row(pos_e)));
    }

    return A;
}
*/


}

#define PRINTM(x) std::cerr << #x << std::endl << x << __FILE__ << ":" << __LINE__ << std::endl << std::endl
#define CHECK_SYMETRIC_POSITIVE(x) Utils::isSymmetricPositiveSemiDefinite(#x, x)

