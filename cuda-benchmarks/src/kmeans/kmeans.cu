#include "kmeans.h"
#include <iostream>
#include "cuda_error_handling.h"
#include "helper_string.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/**
 * Find the closest cluster for a given value.
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

__global__ void kmeans(const int smem_size, const int iterations,
                       const int* input, const int length, 
                       int* cls_cntr, const int k, 
                       int* cls_mean, int* cls_cnt,
                       const int cls_num_partitions,
                       const int num_cls_partitions)
{   
    extern __shared__ int smem[];

    cg::thread_block block = cg::this_thread_block();
    cg::grid_group grid = cg::this_grid();

    unsigned int btr = block.thread_rank();
    unsigned long long gtr = grid.thread_rank();

    // Calculating the mean.
    for (int i = 0; i < num_cls_partitions; i++)
    {
        int index = gtr + i * grid.num_threads();
        if (index < k) 
        {
            cls_mean[index] = 0;
            cls_cnt[index] = 0;
        }
    }
    
    cg::sync(grid);

    // Iterate over the input array.
    for (int i = 0; i < iterations; i++)
    {   
        int value = 0;
        if (gtr + i * grid.num_threads() < length)
            value = input[gtr + i * grid.num_threads()];

        int min_index = 0;
        int min_dist = (value - cls_cntr[min_index])
                     * (value - cls_cntr[min_index]);
        
        int current_cls_part_size = k;

        for (int j = 0; j < cls_num_partitions; j++)
        {
            // Load cluster partition into shared memory.
            for (int l = 0; l <= smem_size / block.num_threads(); l++)
            {
                int index = btr + l * block.num_threads() + j * smem_size;
                if (btr + l * block.num_threads() < smem_size)
                    smem[btr + l * block.num_threads()] = index < k ? cls_cntr[index] : 0;
            }
            
            cg::sync(block);

            // Boundary check.
            if (gtr + i * grid.num_threads() < length)
            {
                // Find closest cluster of this cluster partition.
                int current_min_dist = 0;
                int current_min_index = 0;
                
                find_closest_cluster(value,
                                     smem,
                                     current_cls_part_size < smem_size ? current_cls_part_size : smem_size,
                                     &current_min_index,
                                     &current_min_dist);

                // Compare and update to previous ones.
                if (current_min_dist < min_dist)
                {
                    min_dist = current_min_dist;
                    min_index = current_min_index + j * smem_size;
                }
                current_cls_part_size -= smem_size;
            }
            
            cg::sync(block);
        }

        // Add to cluster with boundary check.
        if (gtr + i * grid.num_threads() < length)
        {
            atomicAdd(&cls_mean[min_index], value);
            atomicAdd(&cls_cnt[min_index], 1);
        }

        cg::sync(block);
    }

    cg::sync(grid);

    // Calculating the mean.
    for (int i = 0; i < num_cls_partitions; i++)
    {
        int index = gtr + i * grid.num_threads();
        if (index < k) 
            cls_mean[index] /= cls_cnt[index];
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
    int* cluster_centers = (int*) malloc(sizeof(int) * k);
    int* cpu_result = (int*) malloc(sizeof(int) * k);
    int* cuda_result = (int*) malloc(sizeof(int) * k);
    int* input = (int*) malloc(sizeof(int) * input_size);
    int* d_input = nullptr;
    int* d_cluster_centers = nullptr;
    int* d_cluster_counter = nullptr;
    int* d_cluster_mean = nullptr;

    std::cout << "Init kmeans\n";
    init_kmeans(input, input_size, cluster_centers, k);
    copy_array(cluster_centers, cpu_result, k);
    std::cout << "Start kmeans_cpu\n";
    kmeans_cpu(cpu_result, k, input, input_size, max_steps);

    std::cout << "Allocating GPU-Ressources\n";
    CUDA_CHECK(cudaMalloc((void**) &d_input, input_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_cluster_centers, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_cluster_counter, k * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**) &d_cluster_mean, k * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_input, input, input_size * sizeof(int), cudaMemcpyHostToDevice));
    
    std::cout << "Calculating remaining parameters\n";
    
    int smem_size_int = smem_size / sizeof(int);

    int input_num_partitions = ceil((double) input_size / (num_blocks * threads_per_block));
    int cls_num_partitions = ceil((double) k / smem_size_int);
    int num_cls_partitions = ceil((double) input_size / (k * threads_per_block));

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
    std::cout << "Input-Partitions: " << input_num_partitions << std::endl;

    std::cout << "Launching Kernel\n";
    // Kernal call.
    while (true)
    {
        CUDA_CHECK(cudaMemcpy(d_cluster_centers, cluster_centers, k * sizeof(int), cudaMemcpyHostToDevice));
    
        CUDA_CHECK(cudaLaunchKernelEx(&config,
                                      kmeans,
                                      smem_size_int, 
                                      input_num_partitions, 
                                      d_input,
                                      input_size,
                                      d_cluster_centers,
                                      k,
                                      d_cluster_mean,
                                      d_cluster_counter,
                                      cls_num_partitions,
                                      num_cls_partitions));
        
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(cuda_result, d_cluster_mean, k * sizeof(int), cudaMemcpyDeviceToHost));

        if (cmp_array(cuda_result, cluster_centers, k))
            break;

        if (max_steps > 0)
            max_steps--;
        if (max_steps == 0)
            break;

        copy_array(cuda_result, cluster_centers, k);  
    }

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
    CUDA_CHECK(cudaFree(d_cluster_centers));
    CUDA_CHECK(cudaFree(d_cluster_counter));
    CUDA_CHECK(cudaFree(d_cluster_mean));
    free(cluster_centers);
    free(cuda_result);
    free(cpu_result);
    free(input);
}