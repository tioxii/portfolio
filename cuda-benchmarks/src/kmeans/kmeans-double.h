#include <stdlib.h>
#include <iostream>
#include <cmath>


/**
 * Check if a given value exists in a array.
 * @param value The value to be checked.
 * @param array The array to be checked.
 * @param length The length of the array.
 * @return True if the values exists in the array, false if not.
 */
bool exists(double value, double* array, int length)
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
void init_kmeans(double* input, const int input_length, double* k_centers, const int k)
{
    srand(time(0));
    for (int i = 0; i < input_length; i++)
        input[i] = rand() % INT_MAX;

    for (int i = 0; i < k; i++)
    {
        int index = rand() % input_length;
        while (exists(input[index], k_centers, i))
            index = rand() % input_length;
        k_centers[i] = input[index];
    }
}

/**
 * Calculates the distance to all cluster centers and returns the index of the closest one.
 * @param value Point from which to calculate the distance.
 * @param cluster_centers Array containing the cluster centers.
 * @param num_clusters Number of clusters.
 * @return Index of the closest cluster.
 */
int find_closest_cluster(const double value, const double* k_centers, const int num_clusters)
{
    // Store current min index and value.
    double min_distance = (value - k_centers[0]) 
                     * (value - k_centers[0]);
    int min_index = 0;

    // Iterate over the current cluster centers.
    for (int j = 1; j < num_clusters; j++)
    {
        // Calculate and update accordingly.
        double distance = (value - k_centers[j]) 
                     * (value - k_centers[j]);
        if (distance < min_distance)
        {
            min_index = j;
            min_distance = distance;
        }
    }
    return min_index;
}

/**
 * Compares two arrays, if the entries are the same. They have to be equal length.
 * @param first_array 
 * @param second_array
 * @param length
 * @return True if the entries are the same, false if not.
 */
bool cmp_array(const double* first_array, const double* second_array, const int length, const double threashhold)
{
    for (int i = 0; i < length; i++)
    {
        if (abs(first_array[i] - second_array[i]) > threashhold)
            return false;
    }
    std::cout << "\nArrays are the same!\n";
    return true;
}

/**
 * Copies the array.
 * @param from Source
 * @param to Destination.
 * @param length Length of the array.
 */
void copy_array(const double* from, double* to, const int length)
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
void kmeans_cpu(double* k_centers, const int num_clusters, const double* input, const int input_length, int max_steps)
{   
    // Allocate helper arrays.
    double* k_mean = (double*) malloc(num_clusters * sizeof(double));
    int* k_count = (int*) malloc(num_clusters * sizeof(int));

    while (true) 
    {
        // Reset values.
        for (int i = 0; i < num_clusters; i++)
        {
            k_mean[i] = 0.0f;
            k_count[i] = 0;
        }

        // Iterate over the entire array.
        for (int i = 0; i < input_length; i++)
        {
            int min_index = find_closest_cluster(input[i], k_centers, num_clusters);
            k_mean[min_index] += input[i];
            k_count[min_index]++;
        }
        // Divide by size of cluster.
        for (int i = 0; i < num_clusters; i++)
            k_mean[i] /= k_count[i];
        // Leave loop when nothing changed
        if (cmp_array(k_centers, k_mean, num_clusters, 0.0f))
            break;

        if (max_steps > 0)
            max_steps--;

        if (max_steps == 0)
            break;
        
        // Update cluster centers
        copy_array(k_mean, k_centers, num_clusters);
    }

    copy_array(k_mean, k_centers, num_clusters);

    // Free ressources.
    free(k_mean);
    free(k_count);
}