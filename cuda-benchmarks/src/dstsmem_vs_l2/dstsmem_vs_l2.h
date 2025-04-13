#include <random>

void init(int* input, const int input_size, int* output, const int output_size)
{
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<int> dist(0, INT_MAX);

    for (int i = 0; i < input_size; i++)
        input[i] = dist(rng);

    for (int i = 0; i < output_size; i++)
        output[i] = 0;
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

void verify_result(int* result, const int size, const int num_blocks)
{
    int errors = 0;
    int iterations = size / num_blocks;

    for (int j = 0; j < iterations; j++)
    {
        int first = result[j * num_blocks];
        for (int i = 1; i < num_blocks; i++)
        {
            if (first != result[i + j * num_blocks])
                errors++;
        }
    }
    std::cout << "Errors: " << errors << std::endl;
}