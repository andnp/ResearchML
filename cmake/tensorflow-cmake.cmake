include(ExternalProject)
ExternalProject_Add(tensorflow-cmake
    GIT_REPOSITORY    https://github.com/andnp/tensorflow-cmake.git
    GIT_TAG           master
    SOURCE_DIR        "${EXTERNAL_DIR}/tensorflow-cmake"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     bash build.sh ${CMAKE_BINARY_DIR}/tensorflow ${EXTERNAL_DIR}/tensorflow
    BUILD_IN_SOURCE   1
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

set(TF_INCLUDE ${EXTERNAL_DIR}/tensorflow/include/google/tensorflow ${EXTERNAL_DIR}/tensorflow/include ${EXTERNAL_DIR}/tensorflow/include/eigen3)
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

set(TF_LIBRARIES ${TF_LIB} ${CUDA_LIB})
