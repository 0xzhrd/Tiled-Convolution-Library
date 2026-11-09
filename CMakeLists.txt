# Tiled convolution library: header only CUDA 2D correlation with arbitrary filter banks
# The library serves as an interface target: headers only, nothing to compile, no dependencies
# Everything built here is a harness: host tests, GPU tests and the benchmarks

# cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
# cmake --build build
# ctest --test-dir build --output-on-failure

# Options:
# T_CONVL_BUILD_CUDA=OFF to disable CUDA tests and benchmarks(host only boundary and reference tests)
# T_CONVL_WITH_OPENCV=ON add the OpenCV-CUDA baseline (needs OpenCV installed with CUDA support built with cudafilters)
# CUDNN_ROOT=<dir> where cuDnn lives(if not in a system path)
# CMAKE_CUDA_ARCHITECTURES defaults to "87:89:90" ; Orin NX, RTX 4060, Hopper

cmake_minimum_required(VERSION 3.24)
project(tiled_convolution VERSION 1.0 LANGUAGES CXX)

option(T_CONVL_BUILD_CUDA "Build GPU tests and the benchmark (needs nvcc)" ON)
option(T_CONVL_WITH_OPENCV "Add the OpenCV-CUDA baseline to the benchmark" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The library: an interface target carrying only the include path and the standard. A consumer wtites target_link_libraries -
# (app PRIVATE tiled_convolution) and gets both

add_Library(tiled_convolution INTERFACE)
add_library(tiled_convolution::tiled_convolution ALIAS tiled_convolution)
target_include_directories(tiled_convolution INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_compile_features(tiled_convolution INTERFACE cxx_std_14)

enable_testing()

# Warnings as errors for our own harnesses, per compiler
function(t_convl_warnings target)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Werror>
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra>)
endfunction()

# Host only tests: plain c++, no CUDA anywhere 

foreach(t test_boundary test_reference)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE tiled_convolution)
    t_convl_warnings(${t})
    add_test(NAME ${t} COMMAND ${t})
endforeach()

if(NOT T_CONVL_BUILD_CUDA)
    message(STATUS "tiled_convolution: host only build (T_CONVL_BUILD_CUDA=OFF)")
    return()
endif()

# CUDA: one fat binary for the Orin(sm_87), the RTX 4060(sm_89) and the Hopper(sm_90)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87 89 90)
endif()
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)

# GPU tests, exit code 77 means "no GPU here": CTest reports it as skipped not passed
foreach(t test_filter_bank test_correctness test_convl2d)
    add_executable(${t} tests/${t}.cu)
    target_include_directories(${t} PRIVATE tests)
    target_link_libraries(${t} PRIVATE tiled_convolution CUDA::cudart)
    t_convl_warnings(${t})
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES SKIP_RETURN_CODE 77)
endforeach()

# Baselines and the benchmark: need cuDNN and NPP. cuDNN is a separate install, without it these two targets are left out with a message

find_path(CUDNN_INCLUDE_DIR cudnn.h
    HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDAToolkit_INCLUDE_DIRS}
    PATH_SUFFIXES include include/x86_64-linux-gnu)
find_library(CUDNN_LIBRARY NAMES cudnn libcudnn.so.9
    HINTS ${CUDNN_ROOT} $ENV{CUDNN_ROOT} ${CUDAToolkit_LIBRARY_DIR}
    PATH_SIFFIXES lib lib64 lib/x86_64-linux-gnu)

if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY OR NOT TARGET CUDA::nppif)
    message(WARNING "tiled_convolution: cuDNN or NPP not found; test_vaselines and bench are not built. "
                    "set CUDNN_ROOT to your cuDNN install.")
    return()
endif()
messgae(STATUS "tiled_convolution: cuDNN ${CUDNN_LIBRARY}")

add_library(t_convl_baselines INTERFACE)
target_include_directories(t_convl_baselines INTERFACE benchmarks/baselines
${CUDNN_INCLUDE_DIR})
target_link_libraries(t_convl_baselines INTERFACE ${CUDNN_LIBRARY} CUDA::nppif CUDA::nppc CUDA::cudart)

if(T_CONVL_WITH_OPENCV)
    find_package(OpenCV REQUIRED COMPONENTS core cudafilters)
    target_compile_definitions(t_convl_baselines INTERFACE T_CONVL_WITH_OPENCV)
    target_link_libraries(t_convl_baslines INTERFACE ${OpenCV_LIBS})
    target_include_directories(t_convl_baselines interface ${OpenCV_INCLUDE_DIRS})
endif()

add_executable(test_baselines tests/test_baselines.cu)
target_include_directories(test_baselines PRIVATE tiled_convolution t_convl_baselines)
t_convl_warnings(test_baselines)
add_test(NAME test_baselines COMMAND test_baselines)
set_tests_properties(test_baselines PROPERTIES SKIP_RETURN_CODE 77)

add_executable(bench benchmarks/bench.cu)
target_link_libraries(bench PRIVATE tiled_convolution t_convl_baselines)
t_convl_warnings(bench)
