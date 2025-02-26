file(GLOB SRCS *.cpp)

add_library(misc ${SRCS})
target_include_directories(
    misc PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(
    misc
    actor
    config
    console
    environment
    network
    utils
)