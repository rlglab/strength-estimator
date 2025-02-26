cmake_minimum_required(VERSION 3.16)

project(strength)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Torch REQUIRED)
find_package(Boost COMPONENTS system thread iostreams)
find_package(ale REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -g -Wall -mpopcnt -O3 -pthread")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wno-unused-function -O0 -pthread")

# for git info
include_directories(${PROJECT_BINARY_DIR}/git_info)

add_subdirectory(strength)
add_subdirectory(strength/misc)
add_subdirectory(strength/trainer)
add_subdirectory(minizero)

string(TOLOWER "${PROJECT_NAME}_${GAME_TYPE}" EXE_FILE_NAME)
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${EXE_FILE_NAME})
