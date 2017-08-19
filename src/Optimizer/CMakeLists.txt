add_library(loss Loss/loss.cpp)
add_library(optimizer_util utils/utils.cpp)

set(OPTIMIZER loss optimizer_util PARENT_SCOPE)
