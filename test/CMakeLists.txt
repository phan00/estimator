find_package(GTest REQUIRED)

message(STATUS "includes: ${GTest_INCLUDES}")
message(STATUS "libraries: ${GTest_LIBRARIES}")

include_directories(${GTest_INCLUDES})

set(sources_list_test
    test.cpp
    test_ekf.cpp
    test_crossover_angle.cpp)

add_executable(test_estimator ${sources_list_test})
target_link_libraries(test_estimator ${GTest_LIBRARIES} pthread armadillo)
target_compile_options(test_estimator PRIVATE -O0 -g3)

include(GoogleTest)
gtest_discover_tests(test_estimator)
