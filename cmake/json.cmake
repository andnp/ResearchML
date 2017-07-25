include(ExternalProject)
ExternalProject_Add(json
    GIT_REPOSITORY    https://github.com/nlohmann/json.git
    GIT_TAG           master
    SOURCE_DIR        "${EXTERNAL_DIR}/json"
    CONFIGURE_COMMAND ""
    BUILD_COMMAND     ""
    INSTALL_COMMAND   ""
    TEST_COMMAND      ""
)
