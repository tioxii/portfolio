#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdio.h>

/**
 * Check if a given value exists in a array.
 * @param value The value to be checked.
 * @param array The array to be checked.
 * @param length The length of the array.
 * @return True if the values exists in the array, false if not.
 */
bool exists(int value, int* array, int length)
{
    for (int i = 0; i < length; i++)
    {
        if (value == array[i])
            return true;
    }
    return false;
}

/**
 * Init array with random values between 0 and 10.
 * Runtime is not deterministic.
 * @param array
 * @param length
 */
void init_kmeans(int* input, const int input_length, int* cluster_centers, const int num_clusters)
{
    srand(time(0));
    for (int i = 0; i < input_length; i++)
        input[i] = rand() % INT_MAX;

    for (int i = 0; i < num_clusters; i++)
    {
        int index = rand() % input_length;
        while (exists(input[index], cluster_centers, i))
            index = rand() % input_length;
        cluster_centers[i] = input[index];
    }
}

/**
 * Print an integer array.
 */
void print_array(const int* array, const int length)
{
    for (int i = 0; i < length; i++)
        printf("%d ", array[i]);
    printf("\n");
}

/**
 * Print an integer array to a file.
 */
void fprint_array(FILE* fptr, int* array, int size)
{
    for (int i = 0; i < size; i++)
        fprintf(fptr, "%d ", array[i]);
    fprintf(fptr, "\n\n");
}

/**
 * Compares two arrays, if the entries are the same. They have to be equal length.
 * @param first_array 
 * @param second_array
 * @param length
 * @return True if the entries are the same, false if not.
 */
bool cmp_array(const int* first_array, const int* second_array, const int length)
{
    for (int i = 0; i < length; i++)
    {
        if (first_array[i] != second_array[i])
            return false;
    }
    std::cout << "\nArrays are the same!\n";
    return true;
}

/**
 * Calculates the distance to all cluster centers and returns the index of the closest one.
 * @param value Point from which to calculate the distance.
 * @param cluster_centers Array containing the cluster centers.
 * @param num_clusters Number of clusters.
 * @return Index of the closest cluster.
 */
int find_closest_cluster_center(const int value, const int* cluster_centers, const int num_clusters)
{
    // Store current min index and value.
    int min_distance = (value - cluster_centers[0]) 
                     * (value - cluster_centers[0]);
    int min_index = 0;

    // Iterate over the current cluster centers.
    for (int j = 1; j < num_clusters; j++)
    {
        // Calculate and update accordingly.
        int distance = (value - cluster_centers[j]) 
                     * (value - cluster_centers[j]);
        if (distance < min_distance)
        {
            min_index = j;
            min_distance = distance;
        }
    }
    return min_index;
}

/**
 * Copies the array.
 * @param from Source
 * @param to Destination.
 * @param length Length of the array.
 */
void copy_array(const int* from, int* to, const int length)
{
    for (int i = 0; i < length; i++)
        to[i] = from[i];
}

/**
 * Performs the kmeans-algorithm from Lloyd.
 * @param cluster_centers Initial cluster centers.
 * @param num_clusters Number of clusters.
 * @param input Input data set.
 * @param input_length Input length.
 */
void kmeans_cpu(int* cluster_centers, const int num_clusters, const int* input, const int input_length, int max_steps)
{   
    // Allocate helper arrays.
    int* cluster_means = (int*) malloc(num_clusters * sizeof(int));
    int* cluster_cnt = (int*) malloc(num_clusters * sizeof(int));

    while (true) 
    {
        // Reset values.
        for (int i = 0; i < num_clusters; i++)
        {
            cluster_means[i] = 0;
            cluster_cnt[i] = 0;
        }
        // Iterate over the entire array.
        for (int i = 0; i < input_length; i++)
        {
            int min_index = find_closest_cluster_center(input[i], cluster_centers, num_clusters);
            cluster_means[min_index] += input[i];
            cluster_cnt[min_index]++;
        }
        // Divide by size of cluster.
        for (int i = 0; i < num_clusters; i++)
            cluster_means[i] /= cluster_cnt[i];
        // Leave loop when nothing changed
        if (cmp_array(cluster_centers, cluster_means, num_clusters))
            break;

        if (max_steps > 0)
            max_steps--;

        if (max_steps == 0)
            break;
        
        // Update cluster centers
        copy_array(cluster_means, cluster_centers, num_clusters);
    }

    copy_array(cluster_means, cluster_centers, num_clusters);

    // Free ressources.
    free(cluster_means);
    free(cluster_cnt);
}

/**
 * Compare CPU-Result with the GPU-Result, if they are the same.
 * Exits the program if they are not the same.
 * @param cpu_result Result from the CPU-Calculation.
 * @param cuda_result Result from the GPU-Calculation.
 * @param length Length of the Results
 */
void cmp_result(const int* cpu_result, const int* cuda_result, const int length)
{
    int errors = 0;
    for (int i = 0; i < length; i++) {
        if (cpu_result[i] != cuda_result[i]) {
            //printf("CPU-Result: %d, CUDA-Result: %d\n", cpu_result[i], cuda_result[i]);
            errors++;
        } else {
            //printf("CPU-Result: %d, CUDA-Result: %d\n", cpu_result[i], cuda_result[i]);
        }
    }
    if (errors == 0)
        printf("kmeans is the same\n");
    else {
        printf("kmeans is not the same! Errors: %d\n", errors);
        exit(EXIT_FAILURE);
    }
}