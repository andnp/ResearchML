INSTALL_DIR=$1

cd ${INSTALL_DIR}

bazel build --config=opt tensorflow:libtensorflow_all.so

# copy the library to the install directory
mkdir -p ${INSTALL_DIR}/lib
cp bazel-bin/tensorflow/libtensorflow_all.so ${INSTALL_DIR}/lib/libtensorflow_all.so

# Copy the source to $INSTALL_DIR/include/google and remove unneeded items:
mkdir -p ${INSTALL_DIR}/include/google/tensorflow
chmod -R 775 ${INSTALL_DIR}/include/google/tensorflow
cp -r tensorflow ${INSTALL_DIR}/include/google/tensorflow/
find ${INSTALL_DIR}/include/google/tensorflow/tensorflow -type f  ! -name "*.h" -delete

# Copy all generated files from bazel-genfiles:
cp  bazel-genfiles/tensorflow/core/framework/*.h ${INSTALL_DIR}/include/google/tensorflow/tensorflow/core/framework
cp  bazel-genfiles/tensorflow/core/lib/core/*.h ${INSTALL_DIR}/include/google/tensorflow/tensorflow/core/lib/core
cp  bazel-genfiles/tensorflow/core/protobuf/*.h ${INSTALL_DIR}/include/google/tensorflow/tensorflow/core/protobuf
cp  bazel-genfiles/tensorflow/core/util/*.h ${INSTALL_DIR}/include/google/tensorflow/tensorflow/core/util
cp  bazel-genfiles/tensorflow/cc/ops/*.h ${INSTALL_DIR}/include/google/tensorflow/tensorflow/cc/ops

# Copy the third party directory:
cp -r third_party ${INSTALL_DIR}/include/google/tensorflow/
rm -r ${INSTALL_DIR}/include/google/tensorflow/third_party/py
