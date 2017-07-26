cmake_minimum_required(VERSION 3.0.0)

project(external NONE)
include(ExternalProject)

# set variables for external dependencies
set(EXTERNAL_DIR "${CMAKE_BINARY_DIR}/external" CACHE PATH "Location where external dependencies will installed")
mark_as_advanced(EXTERNAL_DIR)

# -----
# GTest
# -----

ExternalProject_Add(googletest
  GIT_REPOSITORY    https://github.com/google/googletest.git
  GIT_TAG           master
  SOURCE_DIR        "${CMAKE_BINARY_DIR}/googletest-src"
  BINARY_DIR        "${CMAKE_BINARY_DIR}/googletest-build"
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
    GIT_TAG           master
    SOURCE_DIR        "${EXTERNAL_DIR}/tensorflow"
    CONFIGURE_COMMAND bash configure_tf.sh ${EXTERNAL_DIR}/tensorflow && ./configure
    BUILD_COMMAND     bash install_tf.sh ${EXTERNAL_DIR}/tensorflow
    BUILD_IN_SOURCE   1
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)

ExternalProject_Add_step(tensorflow copy_files
    COMMAND cp {configure,install}_tf.sh ${EXTERNAL_DIR}/tensorflow/
    WORKING_DIRECTORY ${CMAKE_MODULE_PATH}/external/
    DEPENDEES download
    DEPENDERS configure
)

# -----
# Eigen
# -----

ExternalProject_Add(Eigen
        PREFIX ${PROJECT_SOURCE_DIR}/external/eigen
        URL "https://bitbucket.org/eigen/eigen/get/f3a22f35b044.tar.gz"
        DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        BUILD_IN_SOURCE 1
        INSTALL_COMMAND mkdir -p ${EXTERNAL_DIR}/Eigen-headers && cp -R {Eigen,unsupported} ${EXTERNAL_DIR}/Eigen-headers/
)
