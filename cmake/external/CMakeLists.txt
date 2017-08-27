cmake_minimum_required(VERSION 3.0.0)

project(external NONE)
include(ExternalProject)

set(EXTERNAL_DIR ${ROOT_DIR}/external)

# -----
# GTest
# -----

ExternalProject_Add(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           master
  SOURCE_DIR        "${EXTERNAL_DIR}/googletest-src"
  BINARY_DIR        "${EXTERNAL_DIR}/googletest-build"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

# ----
# JSON
# ----

ExternalProject_Add(json
    GIT_REPOSITORY    https://github.com/nlohmann/json.git
    GIT_TAG           master
    SOURCE_DIR        "${EXTERNAL_DIR}/json"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

# ----------
# TensorFlow
# ----------

ExternalProject_Add(tensorflow
    GIT_REPOSITORY    https://github.com/tensorflow/tensorflow.git
    GIT_TAG           v1.3.0-rc0
    SOURCE_DIR        "${EXTERNAL_DIR}/tensorflow"
    CONFIGURE_COMMAND bash configure_tf.sh ${EXTERNAL_DIR}/tensorflow
    BUILD_COMMAND     bash install_tf.sh ${EXTERNAL_DIR}/tensorflow
    BUILD_IN_SOURCE   1
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

ExternalProject_Add_step(tensorflow copy_files
    COMMAND cp configure_tf.sh ${EXTERNAL_DIR}/tensorflow/
    COMMAND cp install_tf.sh ${EXTERNAL_DIR}/tensorflow/
    WORKING_DIRECTORY ${ROOT_DIR}/cmake/external/
    DEPENDEES update
    DEPENDERS configure
)

# -----
# Eigen
# -----

ExternalProject_Add(Eigen
        PREFIX ${EXTERNAL_DIR}
        URL "https://bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz"
        DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
)

# --------
# Protobuf
# --------

ExternalProject_Add(Protobuf
        PREFIX ${EXTERNAL_DIR}
        URL "https://github.com/google/protobuf/archive/0b059a3d8a8f8aa40dde7bea55edca4ec5dfea66.tar.gz"
        DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
        BUILD_IN_SOURCE 1
        CONFIGURE_COMMAND  pwd && ./autogen.sh && ./configure --prefix=${EXTERNAL_DIR}
        BUILD_COMMAND make
        INSTALL_COMMAND make install
)
