#include <cmath>

#ifndef N_FILTER
#define N_FILTER 40
#endif 

#ifndef N_FRAME
#define N_FRAME 121
#endif

namespace MatrixMath {

    void transpose(float** out, float** a, const int a_rows, const int a_cols) {
        for (int i = 0; i < a_rows; i++) {
            for (int j = 0; j < a_cols; j++) {
                out[j][i] = a[i][j];
            }
        }
    };

    float dot_product(float a[], float b[], size_t length) {
        float dot_product = 0.0;
        for (size_t i = 0; i < length; i++) dot_product += a[i] * b[i];
        return dot_product;
    };

    void dot_product(float** prod, float** a, float** b, uint a_rows, uint a_cols, uint b_rows, uint b_cols) {
        for (int r = 0; r < a_rows; r++) {
            for (int l = 0; l < b_cols; l++) prod[r][l] = 0.0;
        }

        for (int i = 0; i < a_rows; ++i) {
            for (int j = 0; j < b_cols; ++j) {
                for (int k = 0; k < a_cols; ++k) {
                    prod[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    };

    template<size_t a_rows, size_t a_cols, size_t b_rows, size_t b_cols> 
    void dot_product(float prod[a_rows][b_cols], float a[a_rows][a_cols], float b[b_rows][b_cols]) {
        for (int r = 0; r < a_rows; r++) {
            for (int l = 0; l < b_cols; l++) prod[r][l] = 0.0;
        }

        for (int i = 0; i < a_rows; ++i) {
            for (int j = 0; j < b_cols; ++j) {
                for (int k = 0; k < a_cols; ++k) {
                    prod[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    };

    template<size_t out_rows, size_t out_cols> 
    void dot_product(float prod[out_rows][out_cols], float** a, float** b, uint a_rows, uint a_cols, uint b_rows, uint b_cols) {
        for (int r = 0; r < a_rows; r++) {
            for (int l = 0; l < b_cols; l++) prod[r][l] = 0.0;
        }

        for (int i = 0; i < a_rows; ++i) {
            for (int j = 0; j < b_cols; ++j) {
                for (int k = 0; k < a_cols; ++k) {
                    prod[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    };

    float cosine_similarity(float a[], float b[], size_t length) {
        float numerator = dot_product(a, b, length);
        float a_norm = std::sqrt(dot_product(a, a, length));
        float b_norm = std::sqrt(dot_product(b, b, length));
        return numerator / (a_norm * b_norm);
    };

};
