#include <cooperative_groups.h>
#include <helper_string.h>
#include <cuda_error_handling.h>
#include "dstsmem_vs_l2.h"

namespace cg = cooperative_groups;

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

__global__ void kernel(const int* input, const int input_size, int* output)
{
    __shared__ int smem[BLOCK_SIZE];

    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();

    unsigned tbc_size = cluster.num_blocks();
    unsigned cbr = cluster.block_rank();

    for (int i = 0; i <= input_size / BLOCK_SIZE; i++)
    {
        int index = block.thread_rank() 
                  + i * BLOCK_SIZE * tbc_size
                  + cbr * BLOCK_SIZE;
        smem[block.thread_rank()]
            = index < input_size ? input[index] : 0;

        cg::sync(cluster);

        const int* dst_smem = cluster.map_shared_rank(smem, 0);
        int value = dst_smem[0];
        
        for (int l = 0; l < tbc_size; l++)
        {
            dst_smem = cluster.map_shared_rank(smem, l);
            for (int j = 1; j < BLOCK_SIZE; j++)
                value = dst_smem[j] < value ? dst_smem[j] : value;
        }
        
        if (block.thread_rank() == 0)
            output[grid.block_rank() + i * grid.num_blocks()] = value; 

        cg::sync(cluster);
    }
}

int main(int argc, char* argv[])
{
    int input_size = 1024;
    if (checkCmdLineFlag(argc, (const char**) argv, "Input_Size"))
        input_size = getCmdLineArgumentInt(argc, (const char**) argv, "Input_Size");

    int num_blocks = 64;
    if (checkCmdLineFlag(argc, (const char**) argv, "Number_of_Blocks"))
        num_blocks = getCmdLineArgumentInt(argc, (const char**) argv, "Number_of_Blocks");

    int tbc_size = 8;
    if (checkCmdLineFlag(argc, (const char**) argv, "Blocks_per_Cluster"))
        tbc_size = getCmdLineArgumentInt(argc, (const char**) argv, "Blocks_per_Cluster");

    if (num_blocks % tbc_size)
    {
        std::cout << "Number of blocks is not divisible by TBC-Size." << std::endl;
        exit(EXIT_FAILURE);
    }

    int output_size = num_blocks * ((input_size / BLOCK_SIZE) + 1);
    
    int* input = (int*) malloc(input_size * sizeof(int));
    int* output = (int*) malloc(output_size * sizeof(int));
    int* d_input = nullptr;
    int* d_output = nullptr;

    CUDA_CHECK(cudaMalloc((void**) &d_input, input_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_output, output_size * sizeof(int)));

    init(input, input_size, output, output_size);
    CUDA_CHECK(cudaMemcpy(d_input, input, input_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output, output, output_size * sizeof(int), cudaMemcpyHostToDevice));

    // Setting launch attributes
    cudaLaunchConfig_t config = {0};
    config.gridDim.x = num_blocks;
    config.blockDim.x = BLOCK_SIZE;

    std::cout << "=== Launch parameters ===" << std::endl;
    std::cout << "Blocks: " << num_blocks << std::endl;
    std::cout << "Input size: " << input_size << std::endl;
    std::cout << "Output size: " << output_size << std::endl;
    std::cout << "Threads per block: " << config.blockDim.x << std::endl;
    std::cout << "Blocks per cluster: " << tbc_size << std::endl;

    CUDA_CHECK(cudaLaunchKernelEx(&config, kernel, d_input, input_size, d_output));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output, d_output, output_size * sizeof(int), cudaMemcpyDeviceToHost));

    std::cout << "=== Results ===" << std::endl;
    verify_result(output, output_size, num_blocks);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(input);
    free(output);
}
