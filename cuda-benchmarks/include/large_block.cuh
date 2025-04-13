#ifndef LARGE_BLOCK
#define LARGE_BLOCK

#include <stdio.h>
#include <iostream>

#define _CG_HAS_CLUSTER_GROUP
#include <cooperative_groups.h>

const int AUTO_TBC_MAX_BLOCK_PER_CLUSTER = 8;

// @param blocks number of blocks
// @param threads number of threads. Can be larger then 1024
// @param smem_size Size of the dynamic shared memory. Can be larger then the
// maximum size per block
// @param max_thread_per_block Internally limit the maximum block size.
template <typename... ExpTypes, typename... ActTypes>
cudaError_t launch_cluster_with_max_thread(void (*kernel)(ExpTypes...),
                                           dim3 blocks,
                                           dim3 threads,
                                           int smem_size,
                                           int max_thread_per_block,
                                           ActTypes &&...args) {
  int total_number_of_threads = threads.x * threads.y * threads.z;

  dim3 cluster(1, 1, 1);

  while (total_number_of_threads > max_thread_per_block) {
    if (threads.x > 2 && threads.x % 2 == 0) {
      threads.x /= 2;
      cluster.x *= 2;
    }

    if (threads.y > 2 && threads.y % 2 == 0) {
      threads.y /= 2;
      cluster.y *= 2;
    }

    if (threads.z > 2 && threads.z % 2 == 0) {
      threads.z /= 2;
      cluster.z *= 2;
    }
    total_number_of_threads = threads.x * threads.y * threads.z;
  }

  int blocks_per_cluster = cluster.x * cluster.y * cluster.z;

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  while (smem_size / blocks_per_cluster > deviceProp.sharedMemPerBlock) {
    if (threads.x > 2 && threads.x % 2 == 0) {
      threads.x /= 2;
      cluster.x *= 2;
    }

    if (threads.y > 2 && threads.y % 2 == 0) {
      threads.y /= 2;
      cluster.y *= 2;
    }

    if (threads.z > 2 && threads.z % 2 == 0) {
      threads.z /= 2;
      cluster.z *= 2;
    }
    blocks_per_cluster = cluster.x * cluster.y * cluster.z;
  }
  smem_size /= blocks_per_cluster;

  if (blocks_per_cluster > AUTO_TBC_MAX_BLOCK_PER_CLUSTER) {
    printf(
        "\n AutoTBC failure: %d Blocks per Cluster required. Maximum is %d\n",
        blocks_per_cluster,
        AUTO_TBC_MAX_BLOCK_PER_CLUSTER);
    exit(EXIT_FAILURE);
  }

  dim3 grid(cluster.x * blocks.x, cluster.y * blocks.y, cluster.z * blocks.z);

  cudaLaunchConfig_t config = {0};
  config.gridDim = grid;
  config.blockDim = threads;
  config.dynamicSmemBytes = smem_size;

  cudaLaunchAttribute attribute[1];
  attribute[0].id = cudaLaunchAttributeClusterDimension;
  attribute[0].val.clusterDim.x = cluster.x;
  attribute[0].val.clusterDim.y = cluster.y;
  attribute[0].val.clusterDim.z = cluster.z;
  config.attrs = attribute;
  config.numAttrs = 1;

  return cudaLaunchKernelEx(&config, kernel, smem_size, args...);
}

template <typename... ExpTypes, typename... ActTypes>
cudaError_t launch_cluster(void (*kernel)(ExpTypes...),
                           dim3 blocks,
                           dim3 threads,
                           int smem,
                           ActTypes &&...args) {
  return launch_cluster_with_max_thread(
      kernel, blocks, threads, smem, 512, args...);
}

template <typename... ExpTypes, typename... ActTypes>
cudaError_t launch_cluster_no_smem(void (*kernel)(ExpTypes...),
                                   dim3 blocks,
                                   dim3 threads,
                                   ActTypes &&...args) {
  return launch_cluster_with_max_thread(
      kernel, blocks, threads, 0, 512, args...);
}

template <typename Type>
class cluster_shmem {
 private:
  Type *m_shmem;
  unsigned short m_size;
  unsigned short m_offset;

 public:
  __device__ cluster_shmem(Type *shmem, int size, int offset = 0) {
    m_shmem = shmem;
    m_size = size / sizeof(Type);
    m_offset = offset;
  }
  __device__ Type &operator[](int index) {
    index += m_offset;
    unsigned rank = index / m_size;
    index = index % m_size;
    Type *dst = cooperative_groups::this_cluster().map_shared_rank(
        &m_shmem[index], rank);
    return *dst;
  }
};

__device__ dim3 operator*(const dim3 &a, const dim3 &b) {
  dim3 result;
  result.x = a.x * b.x;
  result.y = a.y * b.y;
  result.z = a.z * b.z;
  return result;
}

__device__ dim3 operator+(const dim3 &a, const dim3 &b) {
  dim3 result;
  result.x = a.x + b.x;
  result.y = a.y + b.y;
  result.z = a.z + b.z;
  return result;
}

#endif