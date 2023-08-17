pybind11_add_module(${PROJECT_NAME}
    MODULE
        bind_kf.cpp
        bind_ekf.cpp
        bind_models.cpp
        bindings.cpp
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
)

set(FROM_BIN ${CMAKE_CURRENT_BINARY_DIR}/${PYTHON_MODULE_PREFIX}${PROJECT_NAME}${PYTHON_MODULE_EXTENSION})
set(TO_BIN ${CMAKE_CURRENT_SOURCE_DIR}/../scripts)

install(TARGETS ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../scripts
)

find_package(carma CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE armadillo carma::carma)

add_custom_target(install_${PROJECT_NAME}
    COMMAND ${CMAKE_COMMAND} --build . --target install
    DEPENDS ${PROJECT_NAME}
    COMMENT "Installing ${PROJECT_NAME}"
)

add_dependencies(install_${PROJECT_NAME} ${PROJECT_NAME})
