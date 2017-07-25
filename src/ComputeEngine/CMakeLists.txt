add_library(matrix matrix.cpp)
add_library(compute_engine ComputeEngine.cpp)

set(COMPUTE_ENGINE compute_engine matrix PARENT_SCOPE)
