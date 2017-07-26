set(TF_INCLUDE ${EXTERNAL_DIR}/tensorflow/include/google/tensorflow ${EXTERNAL_DIR}/tensorflow/include ${EXTERNAL_DIR}/tensorflow/include/google/tensorflow/third_party/eigen3 ${EXTERNAL_DIR}/Eigen-headers)
find_library(TF_LIB NAMES tensorflow_all
        HINTS
        ${EXTERNAL_DIR}/tensorflow/lib
        /usr/lib
        /usr/local/lib)

find_library(CUDA_LIB NAMES cudnn
        HINTS
        /usr/lib
        /usr/local/cuda/lib64
        /usr/local/lib)

set(TF_LIBRARIES ${TF_LIB})

if (${CUDA_LIB})
        set(TF_LIBRARIES ${TF_LIBRARIES} ${CUDA_LIB})
endif()
