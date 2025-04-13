#include "kmeans.h"
#include "helper_string.h"
#include "cuda_error_handling.h"

#ifndef _CG_HAS_CLUSTER_GROUP
#define _CG_HAS_CLUSTER_GROUP
#endif

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__device__ void print_smem(const int smem_size, const int* smem, const int row_length)
{
    for (int i = 0; i < smem_size; i++)
    {
        if (i % row_length == 0)
            printf("\n");
        printf("%d ", smem[i]);
    }
    printf("\n");
}

__global__ void kmeans_smem(const int smem_size,
                            const int* input,
                            const int length,
                            const int k,
                            const int k_iterations,
                            const int smem_iterations,
                            const int input_iterations,
                            int* k_centers,
                            int* k_mean,
                            int* k_count)
{
    extern __shared__ int smem[];

    cg::thread_block block = cg::this_thread_block();
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();

    unsigned int num_threads = block.num_threads();
    unsigned int btr = block.thread_rank();
    unsigned int tbc_size = cluster.num_blocks();
    unsigned int cbr = cluster.block_rank();
    unsigned long long gtr = grid.thread_rank();

    // Init with 0.
    for (int i = 0; i < k_iterations; i++)
    {
        unsigned long long index = gtr + i * gtr;
        if (index < k)
        {
            k_mean[index] = 0;
            k_count[index] = 0;
        }
    }

    cg::sync(grid);

    // Load into shared memory.
    for (int i = 0; i < smem_iterations; i++)
    {
        unsigned long long index = cbr * smem_size + i * num_threads + btr;
        if (btr + i * num_threads < smem_size)
            smem[btr + i * num_threads] = index < k ? k_centers[index] : 0;
    }

    cg::sync(cluster);

    // Perform k means.
    for (int i = 0; i < input_iterations; i++)
    {
        unsigned long long index = gtr + i * grid.num_threads();
        if (index < length)
        {
            int value = input[index];

            int min_index = 0;
            const int* first = cluster.map_shared_rank(smem, 0);
            int min_distance = (value - *first) * (value - *first);
            int k_left = k;
            
            for (int j = 0; j < tbc_size && k_left > 0; j++)
            {
                const int* dst_smem = cluster.map_shared_rank(smem, j);
                
                for (int i = 0; i < (k_left < smem_size ? k_left : smem_size); i++)
                {
                    int distance = (value - dst_smem[i]) * (value - dst_smem[i]);
                    if (distance < min_distance)
                    {
                        min_distance = distance;
                        min_index = j * smem_size + i;
                    }
                }

                k_left -= smem_size;
            }

            atomicAdd(&k_mean[min_index], value);
            atomicAdd(&k_count[min_index], 1);
        }
    }

    cg::sync(grid);

    for (int i = 0; i < k_iterations; i++)
    {
        unsigned long long index = gtr + i * gtr;
        if (index < k)
            k_mean[index] /= k_count[index];
    }
}

int main(int argc, char* argv[])
{
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    if  (checkCmdLineFlag(argc, (const char**) argv, "help"))
    {
        std::cout << "kmeans: \n"
                  << "--Array_Size\n"
                  << "--Threads_per_Block\n"
                  << "--k\n"
                  << "--Max_Number_of_Steps\n"
                  << "--Shared_Memory_Int\n";
        exit(0);
    }

    int num_blocks = 64;
    int threads_per_block = 1024;
    int k = 3;
    int input_size = 40;
    int smem_size = deviceProp.sharedMemPerBlock;
    int tbc_size = 8;

    if (checkCmdLineFlag(argc, (const char**) argv, "Threads_per_Block"))
        threads_per_block = getCmdLineArgumentInt(argc, (const char**) argv, "Threads_per_Block");

    if (checkCmdLineFlag(argc, (const char**) argv, "SMem_Size_Int"))
        smem_size = getCmdLineArgumentInt(argc, (const char**) argv, "SMem_Size_Int") * sizeof(int);

    if (checkCmdLineFlag(argc, (const char**) argv, "Number_of_Blocks=64"))
        num_blocks = getCmdLineArgumentInt(argc, (const char**) argv, "Number_of_Blocks=64");

    if (checkCmdLineFlag(argc, (const char**) argv, "Input_Size"))
        input_size = getCmdLineArgumentInt(argc, (const char**) argv, "Input_Size");
    
    if (checkCmdLineFlag(argc, (const char**) argv, "k"))
        k = getCmdLineArgumentInt(argc, (const char**) argv, "k");

    if (checkCmdLineFlag(argc, (const char**) argv, "Block_per_Cluster"))
        tbc_size = getCmdLineArgumentInt(argc, (const char**) argv, "Blocks_per_Cluster");

    if (num_blocks == 0 || tbc_size == 0)
    {
        std::cout << "Launch specification don't meet the requirements for kernel launch." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (num_blocks % tbc_size)
    {
        printf("Number of Blocks %d is not divisible by TBC size %d\n", num_blocks, tbc_size);
        exit(EXIT_FAILURE);
    }

    if ((smem_size / sizeof(int)) * tbc_size < k)
    {
        printf("Distributed shared memory is not large enough to host kmeans.\n");
        exit(EXIT_FAILURE);
    }

    // Allocating and init CPU ressources.
    std::cout << "Allocating CPU-Ressources.\n";

    int* k_centers = (int*) malloc(sizeof(int) * k);
    int* cpu_result = (int*) malloc(sizeof(int) * k);
    int* cuda_result = (int*) malloc(sizeof(int) * k);
    int* input = (int*) malloc(sizeof(int) * input_size);

    int* d_input = nullptr;
    int* d_k_centers = nullptr;
    int* d_k_counter = nullptr;
    int* d_k_mean = nullptr;

    init_kmeans(input, input_size, k_centers, k);
    copy_array(k_centers, cpu_result, k);

    // Calculating kmeans on CPU.
    std::cout << "Calculating kmeans on CPU.\n";

    kmeans_cpu(cpu_result, k, input, input_size, 1); //TODO: replace one with max steps.

    // Allocating GPU ressources.
    std::cout << "Allocating GPU-Ressources\n";

    CUDA_CHECK(cudaMalloc((void**) &d_input, input_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_k_centers, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_k_counter, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_k_mean, k * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, input, input_size * sizeof(int), cudaMemcpyHostToDevice));

    // Calculating remaining parameters.
    std::cout << "Calculating remaining parameters\n";

    int smem_size_int = smem_size / sizeof(int);
    
    // The number of partitions the algorithm has to break the input, because there are not enough threads in the grid. 
    int input_iterations = input_size / (num_blocks * threads_per_block);
    input_iterations = input_size % (num_blocks * threads_per_block) ? input_iterations + 1 : input_iterations;
    
    // The number of partitions the algorithm has to break shared_memory, because there are not enough threads.
    int smem_iterations = smem_size_int / threads_per_block;
    smem_iterations = smem_size_int % threads_per_block ? smem_iterations + 1 : smem_iterations;
    
    // The number of partitions the algorithm has to break k, because there are not enough threads in the grid.
    int k_iterations = k / (num_blocks * threads_per_block);
    k_iterations = k % (num_blocks * threads_per_block) ? k_iterations + 1 : k_iterations;

    // Launch configuration.
    std::cout << "Setting launch configuration.\n"; 
    
    cudaLaunchConfig_t config = {0};
    config.gridDim.x = num_blocks;
    config.blockDim.x = threads_per_block;
    config.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute attribute[2];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = tbc_size;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    attribute[1].id = cudaLaunchAttributeCooperative;
    attribute[1].val.cooperative = 1; 

    config.attrs = attribute;
    config.numAttrs = 2;

    std::cout << "Blocks: " << num_blocks << std::endl;
    std::cout << "SMs: " << deviceProp.multiProcessorCount << std::endl;
    std::cout << "TBC-Size: " << tbc_size << std::endl;
    std::cout << "SMem-Size: " << smem_size << std::endl;
    std::cout << "SMem-Size as Int: " << smem_size_int << std::endl;
    std::cout << "k-iterations: " << k_iterations << std::endl;
    std::cout << "SMem-iterations: " << smem_iterations << std::endl;
    std::cout << "Input-iterations: " << input_iterations << std::endl;

    // Launch kernel
    std::cout << "Launching Kernel\n";
    
    CUDA_CHECK(cudaMemcpy(d_k_centers, k_centers, k * sizeof(int), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaLaunchKernelEx(&config,
                                  kmeans_smem,
                                  smem_size_int,
                                  d_input,
                                  input_size,
                                  k,
                                  k_iterations,
                                  smem_iterations,
                                  input_iterations,
                                  d_k_centers,
                                  d_k_mean,
                                  d_k_counter));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(cuda_result, d_k_mean, k * sizeof(int), cudaMemcpyDeviceToHost));

    cmp_result(cpu_result, cuda_result, k);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_k_centers));
    CUDA_CHECK(cudaFree(d_k_counter));
    CUDA_CHECK(cudaFree(d_k_mean));
    free(k_centers);
    free(cuda_result);
    free(cpu_result);
    free(input);
}