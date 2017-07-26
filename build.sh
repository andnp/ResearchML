cores=getconf _NPROCESSORS_ONLN

mkdir -p build
cd build
cmake .. && make -j${cores}
