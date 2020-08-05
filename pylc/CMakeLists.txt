cmake_minimum_required(VERSION 2.8.3)
project(pylc_lib)

include_directories(src/alglib/include/)

file(GLOB alglib_SRC
		"src/alglib/src/*.cpp"
		"src/alglib/include/*.h"
)

add_library(alglib ${alglib_SRC})
set_property(TARGET alglib PROPERTY POSITION_INDEPENDENT_CODE ON)

add_subdirectory(pybind11)

find_package(Boost REQUIRED COMPONENTS system)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR} src/pylc_lib src/alglib/include/)

pybind11_add_module(pylc_lib src/pylc_lib/pylc_lib.cpp)
target_link_libraries(pylc_lib PRIVATE pybind11::module alglib ${OpenCV_LIBRARIES})