file(GLOB SRCS *.cpp)

find_package(pybind11 REQUIRED)
pybind11_add_module(strength_py ${SRCS})
target_link_libraries(
    strength_py
    PUBLIC
    config
    environment
    learner
    utils
    misc
)