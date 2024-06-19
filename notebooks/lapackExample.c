#include <stdio.h>
#include <Python.h>

#include <cblas.h>
int main() {
    double x[3] = {1.0, 2.0, 3.0};
    // Define the vectors
    double y[3] = {4.0, 5.0, 6.0};
    int n = 3; // Length of the vectors

    // Compute the dot product using ddot
    double result = cblas_ddot(n, x, 1, y, 1);

    // Print the result
    printf("Dot product: %f\n", result);

    return 0;
}
