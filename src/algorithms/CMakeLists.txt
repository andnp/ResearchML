add_library(abstract_algorithm abstract_algorithm.cpp abstract_algorithm.hpp)
add_subdirectory(supervised)
add_subdirectory(unsupervised)
add_subdirectory(timeseries)

set(ALGORITHMS ${SUPERVISED_ALGORITHMS} ${UNSUPERVISED_ALGORITHMS} ${TIMESERIES_ALGORITHMS} PARENT_SCOPE)
