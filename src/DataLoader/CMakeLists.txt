add_library(preprocess Preprocess/preprocess.cpp)
add_library(dataloader_utils utils/utils.cpp)

set(DATALOADER preprocess dataloader_utils PARENT_SCOPE)
