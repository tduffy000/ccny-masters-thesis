#include <iostream>
#include "matrix_math.h"

void print_matrix(float** mat, const int rows, const int cols) {
    std::cout << "[";
    for (int i = 0; i < rows; i++) {
        std::cout << "[";
        for (int j = 0; j < cols; j++) {
            std::cout << mat[i][j] << ",";
        }
        std::cout << "]";
        if (i < rows - 1) std::cout << std::endl;
    }
    std::cout << "]" << std::endl;
}

int main() {
    
    // transpose
    const int a_rows = 2;
    const int a_cols = 3;

    float** a = new float *[a_rows];
    for (int i = 0; i < a_rows; i++) a[i] = new float[a_cols];

    for (int i = 0; i < a_rows; i++) {
        for (int j = 0; j < a_cols; j++) a[i][j] = i + j * 1.0;
    }

    float** transposed = new float *[a_cols];
    for (int i = 0; i < a_cols; i++) transposed[i] = new float[a_rows];    

    MatrixMath::transpose(transposed, a, a_rows, a_cols);

    std::cout << "transpose test: " << std::endl;
    print_matrix(a, a_rows, a_cols);
    print_matrix(transposed, a_cols, a_rows);

    // matrix dot product
    const int x_rows = 2;
    const int x_cols = 4;
    const int y_rows = 4;
    const int y_cols = 3;

    float** x = new float *[x_rows];
    for (int i = 0; i < x_rows; i++) x[i] = new float[x_cols];

    for (int i = 0; i < x_rows; i++) {
        for (int j = 0; j < x_cols; j++) x[i][j] = i + j * 1.0;
    }

    float** y = new float *[y_rows];
    for (int i = 0; i < y_rows; i++) y[i] = new float[y_cols];

    for (int i = 0; i < y_rows; i++) {
        for (int j = 0; j < y_cols; j++) y[i][j] = i + j * -1.0;
    }

    float **prod = new float *[x_rows];
    for (int i = 0; i < x_rows; i++) prod[i] = new float[y_cols];

    MatrixMath::dot_product(prod, x, y, x_rows, x_cols, y_rows, y_cols);
    std::cout << "mat multiplication test: " << std::endl;
    std::cout << "x = ";
    print_matrix(x, x_rows, x_cols);
    std::cout << "y = ";
    print_matrix(y, y_rows, y_cols);
    std::cout << "prod = ";
    print_matrix(prod, x_rows, y_cols);

    // vector dot product
    float v1[] = {1.0, 0.0, 0.5};
    float v2[] = {0.5, 0.2, 1.0};

    float dp = MatrixMath::dot_product(v1, v2, 3);
    std::cout << "dot_product test: " << dp << std::endl;

    // cosine similarity
    float cos = MatrixMath::cosine_similarity(v1, v2, 3);
    std::cout << "cosine_similarity test: " << cos << std::endl;

}