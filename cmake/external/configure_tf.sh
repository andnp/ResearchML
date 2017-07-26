linkopts='linkopts = ["-Wl,--version-script=tensorflow/tf_version_script.lds"],'
if [[ `uname` == 'Darwin' ]]; then
    linkopts=""
fi

BUILD_DIR=$1

if [ ! -e ${BUILD_DIR}/.done ]; then

    cd ${BUILD_DIR}

cat <<EOF >> tensorflow/BUILD
# Added build rule
cc_binary(
name = "libtensorflow_all.so",
linkshared = 1,
${linkopts} # if use Mac remove this line
deps = [
    "//tensorflow/cc:cc_ops",
    "//tensorflow/core:framework_internal",
    "//tensorflow/core:tensorflow",
    "//tensorflow/c:c_api",
    "//tensorflow/cc:client_session",
    "//tensorflow/cc:scope"
],
)
EOF
    touch ${BUILD_DIR}/.done
fi
