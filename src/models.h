#pragma once

#include "utils.h"

#include <armadillo>

namespace Models {

template <class M>
M stateModel(const M& x, double T) {
    M F = {{1, T, 0, 0},
           {0, 1, 0, 0},
           {0, 0, 1, T},
           {0, 0, 0, 1}};
    return F*x;
};

template <class M>
M measureModel(const M& x, const M& z = M{}) {
    double angle = atan2(x(2), x(0));
    double range = sqrt(x(0)*x(0) + x(2)*x(2));
    if (!z.empty()) {
        angle = z(0) + Utils::ComputeAngleDifference(angle, z(0));
    }
    M r = {angle, range};
    return trans(r);
};

}
