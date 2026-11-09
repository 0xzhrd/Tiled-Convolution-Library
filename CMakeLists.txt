# Tiled convolution library: header only CUDA 2D correlation with arbitrary filter banks
# The library serves as an interface target: headers only, nothing to compile, no dependencies
# Everything built here is a harness: host tests, GPU tests and the benchmarks

# cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
# cmake --build build
# ctest --test-dir build --output-on-failure

# Options:
# T_CONVL_BUILD_CUDA=OFF to disable CUDA tests and benchmarks(host only boundary and reference tests)
# T_CONVL_WITH_OPENCV=ON add the OpenCV-CUDA baseline (needs OpenCV installed with CUDA support built with cudafilters)
# CUDNN_ROOT=<dir> t