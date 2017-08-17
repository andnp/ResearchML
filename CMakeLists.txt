cmake_minimum_required(VERSION 3.0)

project(gpu)

set(CMAKE_BUILD_TYPE Release)
option(BUILD_TESTS "build tests" OFF)

# set variables for external dependencies
set(EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/external" CACHE PATH "Location where external dependencies will installed")
mark_as_advanced(EXTERNAL_DIR)

# Add modules
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

include(cmake/tensorflow.cmake)
include(cmake/SetCompilerFlags.cmake)

set(PROJECT_LIBRARIES ${TF_LIBRARIES})

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -pedantic")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /opt/local/lib/ /usr/lib64/nvidia/ /usr/local/cuda/lib64)
include_directories(src/ ${PROJECT_INCLUDE_DIRS} ${TF_INCLUDE} ${EXTERNAL_DIR})

add_executable(generate_header generate_header.cpp)

add_custom_command (
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/gpucompute.hpp
  COMMAND generate_header ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS generate_header
)

add_subdirectory(src/algorithms)
add_subdirectory(src/util)
add_subdirectory(src/analysis)
add_subdirectory(src/ComputeEngine)
add_subdirectory(src/DataLoader)
add_subdirectory(src/Optimizer)

add_library(gpucompute gpucompute.cpp gpucompute.hpp)

# Add testing executable
add_executable(tftest tftest.cpp)

# Make sure this only builds when tensorflow is done installing
add_dependencies(tftest gpucompute)
target_link_libraries(gpucompute ${ALGORITHMS} ${OPTIMIZER} ${ANALYSIS} ${DATALOADER} ${COMPUTE_ENGINE} ${UTILS} ${PROJECT_LIBRARIES})
target_link_libraries(tftest gpucompute)

if (BUILD_TESTS)
  include(cmake/gtest.cmake)
  include_directories(.)

  # add_executable(tests tests/tests.cpp)
  # target_link_libraries(tests gtest_main gpucompute)
  add_executable(unit_tests tests/unit_tests.cpp)
  add_dependencies(unit_tests gpucompute)
  target_link_libraries(unit_tests gtest_main gpucompute)
  add_test(NAME unit COMMAND unit_tests)
endif (BUILD_TESTS)
