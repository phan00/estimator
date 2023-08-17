#pragma once

#include <armadillo>

namespace Models {

template <class M>
M stateModel(const M& x, double t) {
    M F = {{1, t, 0, 0},
           {0, 1, 0, 0},
           {0, 0, 1, t},
           {0, 0, 0, 1}};
    return F*x;
};

template <class M>
M measureModel(const M& x) {
    double angle = atan2(x(2), x(0));
    double range = sqrt(x(0)*x(0) + x(2)*x(2));
    M r = {angle, range};
    return trans(r);
};

}
