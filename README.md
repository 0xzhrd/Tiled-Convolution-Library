# Tiled-Convolution-Library

Header only CUDA library for 2D convolution with arbitrary user supplies filter banks; auditable, dependency free, and portable across Jetson Orin(sm_87), Ada(sm_89)
Benchmarked against cuDNN, NPP, and OpenCV-CUDA.

//fill after the sweep, should contain the orin heatmap and then a reading interpretation


## Purpose of this
Some image filters belong outside the neural network, some classical hand-crafted filters are still valuable; especially in real time, safety critical, or resource constrained systems(i.e 
    -Gradient structure tensor: estimating camera motion from image sequences
    -Multi-scale LoG saliency: highlights small targets so the system cna pay attention to them
    -Blur metric: measure how sharp/blurry a frame is and accept or rejct frames before further processing.)
It is required to be explicit(you can see exactly what they are doing), auditable(easy to inspect, verify and debug) and fast on the hardware that flies. Established libraries either do not expose arbitrary filter banks efficiently(cuDNN is built for multi-channel network layers//operations that appear in deep neural networks)

## Correctness tests
Every result is checked and not assumed:
 -test_boundary, test_reference: Host only(CPU) tests. Verifies border handling policies, reference implementation, identity cases etc

 -test_filter_bank: confirms that filter coefficients are copied to the GPU kernel exactly as intended and that different filter slots dont overwrite each other.

 -test_correctness: main correctness suite, runs both library variants against the reference implementation for: 
        (i) filter radii 1 - 15 
        (ii) two tile shapes 
        (iii) fp32 and uint8 input
        (iv) odd sizes
        (v) pitched batches
        (vi) poisoned outputs
        (vii) identity, convention and determinism tests
 
 -test_convl2d: verifies that the runtime dispatcher(function that chooses what kernel to run) produces bit odentical results to calling the kernel directly.

 -test_baselines: compares the library against other established libraries on the cases those libraries suppport.

## Usage
```
#include "t_convl/conv2d.cuh"
#include "t_convl/device_buffer.hpp"   // optional RAII helpers
using namespace tiled_convolution;

FilterBank bank;                                     // bookkeeping for the constant-memory bank
bank.upload(0, Filter("sobel_x", 1, {-1,0,1, -2,0,2, -1,0,1}), stream);

ConvOptions opt;                                      // defaults: Reflect101 border, Auto variant
Variant ran = conv2d(bank, /*slot*/ 0, /*radius*/ 1,
                     in_view, out_view, stream, opt); // asynchronous; returns the variant used
```
    -The caller owns all memory; the library owns only the computation.
    -Input: ImageView<const T> for float, uint8_t (e.g. an NV12 Y plane) or __half, any pitch. Output: ImageView<float>.
    -Convention: correlation (no filter flip), like OpenCV's filter2D. Filter layout row-major, tap (dx, dy) at (dy+r)*(2r+1) + (dx+r). Radius 0–15 (1×1 to 31×31).
    -Border modes: Zero, Clamp (replicate), Reflect101 (mirror without repeating the edge).
    -A .cu file that uploads a filter must be the one that launches   kernels reading it: without -rdc, each .cu file has its own copy of the constant-memory bank.

## Build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release   # add -DCUDNN_ROOT=... if needed
cmake --build build
ctest --test-dir build --output-on-failure                # GPU tests report "skipped" without a GPU
./build/bench --plan                                      # the grid, without a GPU
./build/bench --benchmark --label 4060-wsl2 --out results/4060.csv
python3 scripts/plot_heatmap.py results/4060.csv --out docs/assets

## Hardware targets
CMAKE_CUDA_ARCHITECTURES defaults to 87;89;90: one binary for the Orin NX (deployment), the RTX 4060 (development) and Hopper (cloud validation). The code uses no architecture-specific features, so the deployable path runs unchanged on all three.

## Benchmark notes
-Grid: all 15 odd filter sizes from 3×3 to 31×31 (the specification's "64 random kernel sizes" cannot mean 64 distinct sizes; filter values do not affect timing, so each size uses one fixed-seed random filter), 16 input sizes up to 3840×2160, batch 1/4/8, three borders.
-cuDNN: cudnnConvolutionForward, cross-correlation, FP32 FMA math (TF32 tensor cores disabled for a like-for-like comparison), algorithm chosen by cudnnFindConvolutionForwardAlgorithm. cuDNN 9 marks this API deprecated; it still works; the current interface is the graph API. Zero border only.
-NPP: nppiFilterBorder_32f_C1R_Ctx. NPP's header documents coefficients "stored in reverse order" (true convolution), so the wrapper uploads the flipped filter; replicate border only; no batching, so one call per image is part of its measured cost.
-OpenCV-CUDA: cv::cuda::createLinearFilter; whatever the installed version rejects is recorded as unsupported in the CSV rather than assumed.
-Every runner gets a row in every cell: ok, unsupported, skipped (outside the border contract) or error, with the reason.

## Repository Layout
include/t_convl/      the library (headers only)
tests/                host and GPU tests
benchmarks/           bench.cu and the baseline wrappers
scripts/              plot_heatmap.py
docs/                 crossover.md, figures in docs/assets/
results/              committed benchmark CSVs
docker/               reproducible build

## Roadmap
-Region-of-interest windows (compute a search box, not the whole frame).
-Multi-filter launches: load a tile once, apply N filters (structure tensor, LoG scales).
