# CUDA - Benchmarks
This project contains various implementations of kmeans, using different features of the H100.

## Requirements
In order to run the following application on your system, you'll need:
* [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
* NVIDIA GPU with [Compute Capability](https://developer.nvidia.com/cuda-gpus) 9 or higher.
* [CMake](https://cmake.org/)

## Build
```
cmake -S . -B build
cmake --build build
```

## Additional information
This project also contains some additional benchmarks other than kmeans.
* [Reduction](https://en.wikipedia.org/wiki/Reduction_operator)
* And a test on how well [distributed shared memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#distributed-shared-memory) works.