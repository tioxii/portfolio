#include <cooperative_groups.h>
#include <stdio.h>

#include "large_block.cuh"
#include "cuda_error_handling.h"
#include "helper_string.h"

#include "reduction.h"

namespace cg = cooperative_groups;

__global__ void Reduction(int* input, const int length, const int iterations, int* output)
{
    extern __shared__ int smem[];

    cg::thread_block block = cg::this_thread_block();  
    cg::grid_group grid = cg::this_grid();

    int grid_size = grid.size();
    int block_size = block.size();
    
    int result = 0;
    int index = (int) grid.thread_rank();

    // Sum array with index difference of grid size.
    for (int i = 0; i < iterations; i++)
    {
        if (index < length)
            result += input[index];
        
        index += grid_size;
    }
    
    // Load into shared memory.
    unsigned btr = block.thread_rank();
    smem[btr] = result;
    cg::sync(block);

    // Sum block.
    for (int i = block_size / 2; i > 0; i = i / 2)
    {
        if (btr < i)
            smem[btr] += smem[btr + i];
        cg::sync(block);
    }
    
    // Add final values.
    if (btr == 0)
        atomicAdd(output, smem[0]);
}

int main(int argc, char* argv[])
{
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    if (checkCmdLineFlag(argc, (const char**) argv, "help"))
    {
        std::cout << "reduction: \n"
                  << "--Number_of_Blocks\n"
                  << "--Array_Size\n"
                  << "--Threads_per_Block\n";
        exit(0);
    }

    // Initialisation
    int blocks = 64;
    int input_size = 1000000;
    int threads_per_block = 1024;

    if (checkCmdLineFlag(argc, (const char**) argv, "Threads_per_Block"))
        threads_per_block = getCmdLineArgumentInt(argc, (const char**) argv, "Threads_per_Block");

    int smem_size = threads_per_block * sizeof(int);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, Reduction, threads_per_block, smem_size);
    blocks *= deviceProp.multiProcessorCount;

    if (checkCmdLineFlag(argc, (const char**) argv, "Number_of_Blocks"))
        blocks = getCmdLineArgumentInt(argc, (const char**) argv, "Number_of_Blocks");
    
    if (checkCmdLineFlag(argc, (const char**) argv, "Array_Size"))
        input_size = getCmdLineArgumentInt(argc, (const char**) argv, "Array_Size");

    // Allocate memory, init variables and calculate cpu result.
    int* input = (int*) malloc(sizeof(int) * input_size);
    int* d_input = nullptr;
    int* d_output = nullptr;

    init_array_random(input, input_size);
    int cpu_result = reduction_cpu(input, input_size);

    CUDA_CHECK(cudaMalloc((void**) &d_input, input_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_output, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, input, input_size * sizeof(int), cudaMemcpyHostToDevice));

    int iterations = (input_size / (threads_per_block * blocks)) + 1;

    std::cout << "Blocks: " << blocks << "\n"
              << "Threads per Block: " << threads_per_block << "\n"
              << "Shared Memory size: " << smem_size << "\n"
              << "Iterations: " << iterations << "\n"
              << "Length: " << input_size << "\n";

    // CUDA kernal call.
    Reduction<<<blocks, threads_per_block, smem_size>>>(d_input, input_size, iterations, d_output);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back memory and compare results.
    int cuda_result = 0;
    CUDA_CHECK(cudaMemcpy(&cuda_result, d_output, sizeof(int), cudaMemcpyDeviceToHost));

    cmp_results(&cpu_result, &cuda_result, 1);
    
    // Free ressources.
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaFree(d_input));
    free(input);
}