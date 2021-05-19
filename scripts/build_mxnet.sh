source arch_build.sh

# only download mxnet if it has not already been downloaded
if [ ! -d "./mxnet" ]
then
    git clone https://github.com/apache/incubator-mxnet ./mxnet
    cd mxnet
    git checkout tags/1.8.0
    git submodule update --init --recursive
    cd ..
fi

if [ "$build_type" = "nist" ]; then
	#########
	# build for CPU in single thread mode and MKL backend
	#########
	echo "Building mxnet for CPU, single thread mode, MKL backend"

  cd mxnet

  # if the build dir exists, it must have been restored from cache
  # no need to rebuild it
  if [ -d "./build_nist" ]
  then
      exit 0
  fi

	mkdir build_nist
	cd build_nist

  # Intel MKL must be installed to /opt/intel/compilers_and_libraries/linux [default install location]
  cmake -DUSE_CPP_PACKAGE=1 -DBUILD_CPP_EXAMPLES=OFF -DUSE_CUDA=0 -DUSE_MKL_IF_AVAILABLE=1 -DUSE_BLAS=mkl -DUSE_OPENCV=0 -DUSE_LAPACK=0 -DUSE_OPENMP=0 \
  -DMKL_INCLUDE_DIR=/opt/intel/compilers_and_libraries/linux/mkl/include -DMKL_RT_LIBRARY=/opt/intel/compilers_and_libraries/linux/mkl/lib/intel64/libmkl_rt.so -DCMAKE_BUILD_TYPE=Release -D CMAKE_CXX_FLAGS="-march=broadwell" ..

	# multithreaded make
  make -j4

	make DESTDIR=./packaged install

elif [ "$build_type" = "cuda_10" ]; then
	#########
	# build with CUDA 10 support
	#########

	cd mxnet
	# if the build dir exists, it must have been restored from cache
  # no need to rebuild it
  if [ -d "./build_cuda_10" ]
  then
    exit 0
  fi

  mkdir build_cuda_10
  cd build_cuda_10

  cmake\
 -DBLAS=open\
 -DUSE_CUDA=ON\
 -DUSE_CUDNN=ON\
 -DMXNET_CUDA_ARCH="5.2;6.0;6.1;7.0;7.5"\
 -DENABLE_CUDA_RTC=OFF\
 -DCMAKE_BUILD_TYPE=Release\
 -DUSE_F16C=OFF\
 -GNinja\
 -DUSE_LAPACK=OFF\
 -DUSE_JEMALLOC=OFF\
 -DUSE_CPP_PACKAGE=ON\
 -DUSE_SIGNAL_HANDLER=OFF\
 -DUSE_OPENCV=OFF\
 -DUSE_MKL_IF_AVAILABLE=OFF\
 -DUSE_MKLDNN=OFF\
 -DBUILD_CPP_EXAMPLES=OFF\
 -DCMAKE_INSTALL_PREFIX=./packaged\
 ..

 cmake --build . -j 5
 cmake --build . --target install

elif [ "$build_type" = "cuda_11" ]; then
	#########
	# build with CUDA support
	#########

	cd mxnet
	# if the build dir exists, it must have been restored from cache
  # no need to rebuild it
  if [ -d "./build_cuda_11" ]
  then
    exit 0
  fi

  mkdir build_cuda_11
  cd build_cuda_11

 cmake\
 -DBLAS=open\
 -DUSE_CUDA=ON\
 -DUSE_CUDNN=ON\
 -DMXNET_CUDA_ARCH="5.2;6.0;6.1;7.0;7.5 8.0 8.6"\
 -DENABLE_CUDA_RTC=OFF\
 -DCMAKE_BUILD_TYPE=Release\
 -DUSE_F16C=OFF\
 -GNinja\
 -DUSE_LAPACK=OFF\
 -DUSE_JEMALLOC=OFF\
 -DUSE_CPP_PACKAGE=ON\
 -DUSE_SIGNAL_HANDLER=OFF\
 -DUSE_OPENCV=OFF\
 -DUSE_MKL_IF_AVAILABLE=OFF\
 -DUSE_MKLDNN=OFF\
 -DBUILD_CPP_EXAMPLES=OFF\
 -DCMAKE_INSTALL_PREFIX=./packaged\
 ..

 cmake --build . -j 5
 cmake --build . --target install

elif [ "$build_type" = "aarch64_cuda" ]; then
	#########
	# build with CUDA 10 support
	#########

	cd mxnet
	# if the build dir exists, it must have been restored from cache
  # no need to rebuild it
  if [ -d "./build_aarch64_cuda" ]
  then
    exit 0
  fi

  mkdir build_aarch64_cuda
  cd build_aarch64_cuda

  # Need to delete line 15 from cpp-package/CMakeLists.txt
  # See: https://github.com/apache/incubator-mxnet/issues/20222
  # We then copy the op.h from our own repo

  sed -i '15d' ../cpp-package/CMakeLists.txt

 # See c-sdks/patches dir for more info
  cp ../../../patches/op.h ../cpp-package/include/mxnet-cpp

# CMAKE_TOOLCHAIN_FILE is set in docker image.
  cmake\
 -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} \
 -DUSE_OPENMP=ON \
 -DUSE_BLAS=Open \
 -DUSE_CUDA=ON\
 -DUSE_CUDNN=ON\
 -DMXNET_CUDA_ARCH="5.3;6.2;7.2"\
 -DENABLE_CUDA_RTC=OFF\
 -DCMAKE_BUILD_TYPE=Release\
 -DUSE_F16C=OFF\
 -GNinja\
 -DUSE_LAPACK=OFF\
 -DUSE_JEMALLOC=OFF\
 -DUSE_CPP_PACKAGE=ON\
 -DUSE_SIGNAL_HANDLER=OFF\
 -DUSE_OPENCV=OFF\
 -DUSE_MKL_IF_AVAILABLE=OFF\
 -DUSE_MKLDNN=OFF\
 -DBUILD_CPP_EXAMPLES=OFF\
 -DCMAKE_INSTALL_PREFIX=./packaged\
 ..

 cmake --build . -j 5
 cmake --build . --target install

 # cp ../../../patches/op.h ./packaged/include/mxnet-cpp/

else
  echo "unsupported build type: $build_type"
  exit 1
fi;

