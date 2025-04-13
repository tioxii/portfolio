#include <cstdlib>
#include <ctime>
#include <iostream>

/**
 * Init array with random values between 0 and 10.
 * @param array
 * @param length
 */
void init_array_random(int* array, const int length)
{
    srand(time(0));
    for (int i = 0; i < length; i++)
        array[i] = rand() % 10 + 1;
}

int reduction_cpu(const int* input, const int length)
{
    int result = 0;
    for (int i = 0; i < length; i++)
        result += input[i];
    return result;
}

void cmp_results(const int* result_cpu, const int* result_gpu, const int length) 
{
    int errors = 0;
    for (int i = 0; i < length; i++) {
        if (result_cpu[i] != result_gpu[i]) {
            errors++;
        }
    }
    if (errors == 0)
        printf("Reduction is the same\n");
    else {
        printf("Reduction is not the same! Errors: %d\n", errors);
        exit(EXIT_FAILURE);
    }
}

void printIntArray(const int* array, const int length)
{
    printf("Array: ");
    for (int i = 0; i < length; i++)
        printf("%d ", array[i]);
    printf("\n");
}