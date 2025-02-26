file(GLOB SRCS *.cpp)
add_executable(strength ${SRCS})
target_link_libraries(
    strength
    config
    console
    environment
    network
    utils
    misc
    ${Boost_LIBRARIES}
    ${TORCH_LIBRARIES}
)