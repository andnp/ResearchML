set(TF_INCLUDE  ${EXTERNAL_DIR}/tensorflow/include/google/tensorflow
                ${EXTERNAL_DIR}/tensorflow/include
                ${EXTERNAL_DIR}/src/Eigen
                ${EXTERNAL_DIR}/src/nsync/public
                ${EXTERNAL_DIR}/include)
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
