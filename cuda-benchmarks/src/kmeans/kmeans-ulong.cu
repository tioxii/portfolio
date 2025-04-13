#include "kmeans-ulong.h"
#include <iostream>
#include "cuda_error_handling.h"
#include "helper_string.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
using ull = unsigned long long;

/**
 * Find the closest cluster for a given value.d
 * @param value The value for which the closest cluster is determined.
 * @param cluster_centers The cluster centers.
 * @param k The number of clusters.
 * @param index The variable in which the index of the closest cluster is returned.
 */
__device__ void find_closest_cluster(const int value, 
                                     const int* cluster_centers, 
                                     const int k,
                                     int* index,
                                     int* distance)
{
    int min_distance = (value - cluster_centers[0])
                     * (value - cluster_centers[0]);
    int min_index = 0;

    for (int j = 1; j < k; j++)
    {
        int distance = (value - cluster_centers[j])
                     * (value - cluster_centers[j]);
        if (distance < min_distance)
        {
            min_index = j;
            min_distance = distance;
        }
    }
    *distance = min_distance;
    *index = min_index;
}

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

/**
 * Loads the cluster centers of into shared memory.
 * @param smem Shared memory.
 * @param smem_size Size of shared memory.
 * @param k_centers Global memory array containing the k_centers.
 * @param k Number of cluster centers.
 * @param j Which partition to load from k.
 */
__device__ void load_k_smem(int* smem, const int smem_size, int* k_centers, const int k, const int j)
{
    cg::thread_block block = cg::this_thread_block();
    unsigned btr = block.thread_rank();
    for (int l = 0; l <= smem_size / block.num_threads(); l++)
    {
        int index = btr + l * block.num_threads() + j * smem_size;
        if (btr + l * block.num_threads() < smem_size)
            smem[btr + l * block.num_threads()] = index < k ? k_centers[index] : 0;
    }
}

__global__ void kmeans(const int smem_size, 
                       const int input_iterations,
                       const int* input, 
                       const int input_length, 
                       int* k_centers, 
                       const int k, 
                       ull* k_mean, 
                       int* k_count,
                       const int k_smem_iterations,
                       const int k_iterations)
{   
    extern __shared__ int smem[];

    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    // Setting values to 0.
    for (int i = 0; i < k_iterations; i++)
    {
        ull index = grid.thread_rank() + i 
                  * grid.num_threads();
        if (index < k) 
        {
            k_mean[index] = 0;
            k_count[index] = 0;
        }
    }

    cg::sync(grid);

    // Iterate over the input array.
    for (int i = 0; i < input_iterations; i++)
    {   
        int value = 0;
        if (grid.thread_rank() + i * grid.num_threads() < input_length)
            value = input[grid.thread_rank() + i * grid.num_threads()];

        int min_index = 0;
        int min_dist = (value - k_centers[min_index])
                     * (value - k_centers[min_index]);
        
        int k_left = k;

        for (int j = 0; j < k_smem_iterations; j++)
        {
            // Load cluster partition into shared memory.
            load_k_smem(smem, smem_size, k_centers, k, j);
            
            cg::sync(block);

            // Boundary check.
            if (grid.thread_rank() + i * grid.num_threads() < input_length)
            {
                // Find closest cluster of this cluster partition.
                int current_min_dist = 0;
                int current_min_index = 0;
                
                find_closest_cluster(value,
                                     smem,
                                     k_left < smem_size ? k_left : smem_size,
                                     &current_min_index,
                                     &current_min_dist);

                // Compare and update to previous ones.
                if (current_min_dist < min_dist)
                {
                    min_dist = current_min_dist;
                    min_index = current_min_index + j * smem_size;
                }
                k_left -= smem_size;
            }
            
            cg::sync(block);
        }

        // Add to cluster with boundary check.
        if (grid.thread_rank() + i * grid.num_threads() < input_length)
        {
            atomicAdd(&k_mean[min_index], value);
            atomicAdd(&k_count[min_index], 1);
        }

        cg::sync(block);
    }

    cg::sync(grid);

    // Calculating the mean.
    for (int i = 0; i <= k / grid.num_threads(); i++)
    {
        ull index = grid.thread_rank() + i 
                  * grid.num_threads();
        if (index < k) 
            k_mean[index] /= k_count[index];
    }
}

int main(int argc, char* argv[])
{   
    cudaDeviceProp deviceProp;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

    if (checkCmdLineFlag(argc, (const char**) argv, "help"))
    {
        std::cout << "kmeans: \n"
                  << "--Array_Size\n"
                  << "--Threads_per_Block\n"
                  << "--k\n"
                  << "--Max_Number_of_Steps\n"
                  << "--Shared_Memory_Int\n";
        exit(0);
    }

    int num_blocks = 1;
    int threads_per_block = min(deviceProp.maxThreadsPerBlock, 
                                deviceProp.maxThreadsPerMultiProcessor);
    int k = 3;
    int input_size = 40;
    int max_steps = -1;
    int smem_size = deviceProp.sharedMemPerBlock;
    bool random_init = false;

    // Reading command line flags.
    if (checkCmdLineFlag(argc, (const char**) argv, "Threads_per_Block"))
        threads_per_block = getCmdLineArgumentInt(argc, (const char**) argv, "Threads_per_Block");

    if (checkCmdLineFlag(argc, (const char**) argv, "Shared_Memory_Int"))
        smem_size = getCmdLineArgumentInt(argc, (const char**) argv, "Shared_Memory_Int") * sizeof(int);

    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, kmeans, threads_per_block, smem_size);
    num_blocks *= deviceProp.multiProcessorCount;

    if (checkCmdLineFlag(argc, (const char**) argv, "Number_of_Blocks"))
        num_blocks = getCmdLineArgumentInt(argc, (const char**) argv, "Number_of_Blocks");
    
    if (checkCmdLineFlag(argc, (const char**) argv, "Array_Size"))
        input_size = getCmdLineArgumentInt(argc, (const char**) argv, "Array_Size");

    if (checkCmdLineFlag(argc, (const char**) argv, "k"))
        k = getCmdLineArgumentInt(argc, (const char**) argv, "k");

    if (checkCmdLineFlag(argc, (const char**) argv, "Max_Number_of_Steps"))
        max_steps = getCmdLineArgumentInt(argc, (const char**) argv, "Max_Number_of_Steps");

    if (checkCmdLineFlag(argc, (const char**) argv, "Random_Init"))
        random_init = true;

    if (num_blocks == 0)
    {
        std::cout << "Launch specification don't meet the requirements for cooperative kernel launch." << std::endl;
        exit(EXIT_FAILURE);
    }

    if (k > input_size)
    {
        std::cout << "Too many clusters in order to initialize properly." << std::endl;
        exit(EXIT_FAILURE);
    }

    std::cout << "Allocating CPU-Ressources.\n";
    // Allocating and init ressources.
    int* k_centers = (int*) malloc(sizeof(int) * k);
    int* cpu_result = (int*) malloc(sizeof(int) * k);
    int* cuda_result = (int*) malloc(sizeof(int) * k);
    ull* k_mean = (ull*) malloc(sizeof(ull) * k);
    int* input = (int*) malloc(sizeof(int) * input_size);
    int* d_input = nullptr;
    int* d_k_centers = nullptr;
    int* d_k_count = nullptr;
    ull* d_k_mean = nullptr;

    std::cout << "Init kmeans\n";
    init_kmeans(input, input_size, k_centers, k, random_init);
    copy_array(k_centers, cpu_result, k);
    std::cout << "Start kmeans_cpu\n";
    kmeans_cpu(cpu_result, k, input, input_size, max_steps);

    std::cout << "Allocating GPU-Ressources\n";
    CUDA_CHECK(cudaMalloc((void**) &d_input, input_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_k_centers, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_k_count, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_k_mean, k * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_input, input, input_size * sizeof(int), cudaMemcpyHostToDevice));
    
    std::cout << "Calculating remaining parameters\n";
    
    int smem_size_int = smem_size / sizeof(int);

    // The number of partitions the algorithm has to break the input, because there are not enough threads in the grid. 
    int input_iterations = input_size / (num_blocks * threads_per_block);
    input_iterations = input_size % (num_blocks * threads_per_block) ? input_iterations + 1 : input_iterations;

    // The number of partitions the algorithm has to break k, because it doesn't fit into shared_memory.
    int k_smem_iterations =  k / (smem_size_int);
    k_smem_iterations = k % (smem_size_int) ? k_smem_iterations + 1 : k_smem_iterations;

    // The number of partitions the algorithm has to break k, because there are not enough threads in the grid.
    int k_iterations = k / (num_blocks * threads_per_block);
    k_iterations = k % (num_blocks * threads_per_block) ? k_iterations + 1 : k_iterations;

    cudaLaunchConfig_t config = {0};
    config.gridDim.x = num_blocks;
    config.blockDim.x = threads_per_block;
    config.dynamicSmemBytes = smem_size;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeCooperative;
    attribute[0].val.cooperative = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    std::cout << "Input-Size: " << input_size << std::endl;
    std::cout << "Number of Blocks: " << num_blocks << std::endl;
    std::cout << "Threads per Block: " << threads_per_block << std::endl;
    std::cout << "Shared-Memory per Block: " << smem_size << std::endl;
    std::cout << "Shared-Memory per Block (int): " << smem_size_int << std::endl;
    std::cout << "Input-Partitions: " << input_iterations << std::endl;

    std::cout << "Launching Kernel\n";
    // Kernal call.
    while (true)
    {
        CUDA_CHECK(cudaMemcpy(d_k_centers, k_centers, k * sizeof(int), cudaMemcpyHostToDevice));
    
        CUDA_CHECK(cudaLaunchKernelEx(&config,
                                      kmeans,
                                      smem_size_int, 
                                      input_iterations, 
                                      d_input,
                                      input_size,
                                      d_k_centers,
                                      k,
                                      d_k_mean,
                                      d_k_count,
                                      k_smem_iterations,
                                      k_iterations));
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(k_mean, d_k_mean, k * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

        if (cmp_array(k_mean, k_centers, k))
            break;

        if (max_steps > 0)
            max_steps--;
        if (max_steps == 0)
            break;

        copy_array(k_mean, k_centers, k);  
    }

    copy_array(k_mean, cuda_result, k);

    if (input_size < 20)
    {
        std::cout << "CPU-Result: \n";
        print_array(cpu_result, k);
        std::cout << "CUDA-Result: \n";
        print_array(cuda_result, k);
    }

    // Compare results.
    cmp_result(cpu_result, cuda_result, k);

    // Free ressources.
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_k_centers));
    CUDA_CHECK(cudaFree(d_k_count));
    CUDA_CHECK(cudaFree(d_k_mean));
    free(k_centers);
    free(cuda_result);
    free(cpu_result);
    free(input);
    free(k_mean);
}