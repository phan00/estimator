
#include "utils.h"

#include <gtest/gtest.h>

TEST (TestCrossover, ComputeAngleDifference) {

    struct AngleDifferenceTest {
        double Angle1;
        double Angle2;
        double Expected;

        void runTest() {
            double computeDelta = Utils::ComputeAngleDifference(Angle1, Angle2);
            ASSERT_NEAR(Expected, computeDelta, 1e-10);
        }
    };

    double expectedDelta = M_PI * 1e-2;

    AngleDifferenceTest tests[] = {
        {-M_PI + expectedDelta/2, M_PI - expectedDelta/2, expectedDelta},
        { M_PI, -M_PI - expectedDelta, expectedDelta},
        {-M_PI, -M_PI + expectedDelta, -expectedDelta},
        {0, expectedDelta, -expectedDelta}
    };

    for (auto t : tests) {
        t.runTest();
    }
}
