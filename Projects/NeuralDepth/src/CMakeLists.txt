cmake_minimum_required(VERSION 3.10.2)

project(neural_depth)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIRS})

include_directories("./struct")

# pybind11_add_module(nerual_depth code.cpp)
add_executable(temp code.cpp struct/mesh.cpp)
target_link_libraries(temp ${OpenCV_LIBS})
