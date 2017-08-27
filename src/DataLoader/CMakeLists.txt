add_library(preprocess Preprocess/preprocess.cpp)
add_library(dataloader_utils utils/utils.cpp)

set(DATALOADER dataloader_utils preprocess PARENT_SCOPE)
