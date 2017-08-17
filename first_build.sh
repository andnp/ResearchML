cores=`getconf _NPROCESSORS_ONLN`

mkdir -p build/external
cp cmake/external/CMakeLists.txt build/external/
cd build/external

root="$(cd "$(dirname "../..")"; pwd)/$(basename "../..")"

cmake . -DROOT_DIR=${root} && make

cd ..
cmake ..
make -j${cores}
