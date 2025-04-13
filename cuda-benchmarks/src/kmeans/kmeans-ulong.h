#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdio.h>
#include <set>
#include <random>
#include <algorithm>

/**
 * Init array with random values between 0 and 10.
 * Runtime is not deterministic.
 * @param input
 * @param length_length
 * @param k_centers
 * @param k
 */
void init_kmeans_random(int* input, const int input_length, int* k_centers, const int k)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<int> dist(0, INT_MAX);
    std::set<int> random_number_set;

    for(int i = 0; random_number_set.size() < input_length && i < input_length * 100; i++)
        random_number_set.insert(dist(rng));

    if (random_number_set.size() < input_length)
    {
        std::cout << "Failed to initialize input" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::copy(random_number_set.begin(), random_number_set.end(), input);
    std::vector<int> random_number_vc(random_number_set.begin(), random_number_set.end());

    for(int i = 0; i < k; i++)
    {
        int index = dist(rng) % random_number_vc.size();
        k_centers[i] = random_number_vc[index];
        random_number_vc.erase(random_number_vc.begin() + index);
    }
}

/**
 * Init kmeans deterministicly, but randomly shuffels the entries.
 * The input numbers are not random, only the order is.
 * @param input
 * @param length_length
 * @param k_centers
 * @param k
 */
void init_kmeans_deterministic(int* input, const int input_length, int* k_centers, const int k)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<int> dist(0, INT_MAX);
    std::set<int> random_number_set;

    if (input_length > INT_MAX)
    {
        std::cout << "Failed to initialize input: Input array is too large." << std::endl;
        exit(EXIT_FAILURE);
    }

    for(int i = 0; random_number_set.size() < input_length; i++)
        random_number_set.insert(i + 1);

    if (random_number_set.size() < input_length)
    {
        std::cout << "Failed to initialize input" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::copy(random_number_set.begin(), random_number_set.end(), input);
    std::vector<int> random_number_vc(random_number_set.begin(), random_number_set.end());

    for(int i = 0; i < k; i++)
    {
        int index = dist(rng) % random_number_vc.size();
        k_centers[i] = random_number_vc[index];
        random_number_vc.erase(random_number_vc.begin() + index);
    }
}

/**
 * Init kmeans.
 * @param input
 * @param length_length
 * @param k_centers
 * @param k
 * @param random If array should be initialized randomly.
 */
void init_kmeans(int* input, const int input_length, int* k_centers, const int k, bool random = true)
{
    if (random)
        init_kmeans_random(input, input_length, k_centers, k);
    else
        init_kmeans_deterministic(input, input_length, k_centers, k);
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
bool cmp_array(const unsigned long long* first_array, const int* second_array, const int length)
{
    for (int i = 0; i < length; i++)
    {
        if (((int) first_array[i]) != second_array[i])
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
int find_closest_cluster_center(const int value, const int* k_centers, const int k)
{
    // Store current min index and value.
    int min_distance = (value - k_centers[0]) 
                     * (value - k_centers[0]);
    int min_index = 0;

    // Iterate over the current cluster centers.
    for (int j = 1; j < k; j++)
    {
        // Calculate and update accordingly.
        int distance = (value - k_centers[j]) 
                     * (value - k_centers[j]);
        if (distance < min_distance)
        {
            min_index = j;
            min_distance = distance;
        }
    }
    return min_index;
}

void copy_array(const unsigned long long* from, int* to, const int length)
{
    for (int i = 0; i < length; i++)
        to[i] = (int) from[i];
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
void kmeans_cpu(int* k_centers, const int k, const int* input, const int input_length, int max_steps)
{   
    // Allocate helper arrays.
    unsigned long long* k_mean = (unsigned long long*) malloc(k * sizeof(unsigned long long));
    int* k_count = (int*) malloc(k * sizeof(int));

    while (true) 
    {
        // Reset values.
        for (int i = 0; i < k; i++)
        {
            k_mean[i] = 0;
            k_count[i] = 0;
        }
        // Iterate over the entire array.
        for (int i = 0; i < input_length; i++)
        {
            int min_index = find_closest_cluster_center(input[i], k_centers, k);
            k_mean[min_index] += input[i];
            k_count[min_index]++;
        }
        // Divide by size of cluster.
        for (int i = 0; i < k; i++)
            k_mean[i] /= k_count[i];
        // Leave loop when nothing changed
        if (cmp_array(k_mean, k_centers, k))
            break;

        if (max_steps > 0)
            max_steps--;

        if (max_steps == 0)
            break;
        
        // Update cluster centers
        copy_array(k_mean, k_centers, k);
    }

    copy_array(k_mean, k_centers, k);

    // Free ressources.
    free(k_mean);
    free(k_count);
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
        if (cpu_result[i] != cuda_result[i])
            errors++;
        if (length < 10)
            printf("CPU-Result: %d, CUDA-Result: %d\n", cpu_result[i], cuda_result[i]);
    }
    if (errors == 0)
        printf("kmeans is the same\n");
    else {
        printf("kmeans is not the same! Errors: %d\n", errors);
        exit(EXIT_FAILURE);
    }
}