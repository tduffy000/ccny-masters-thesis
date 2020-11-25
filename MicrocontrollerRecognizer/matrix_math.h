namespace MatrixMath {

    void transpose(float ** out, float ** a, int rows, int cols) {
        for (int i = 0; i < cols; i++) {
            float* t_row = new float[rows];
            for (int j = 0; j < rows; j++) {
                t_row[j] = a[j][i];
            }
            out[i] = t_row;
        }
    }

    void dot_product(float** out, float ** a, float ** b, int a_rows, int a_cols, int b_rows, int b_cols) {
        for (int r = 0; r < a_rows; r++) {
            float* row = new float[b_cols];
            for (int l = 0; l < b_cols; l++) row[l] = 0.0;
            out[r] = row;
        }

        for (int i = 0; i < a_rows; ++i) {
            for (int j = 0; j < b_cols; ++j) {
                for (int k = 0; k < a_cols; ++k) {
                    out[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }

    float dot_product(float a[], float b[], size_t length) {
        float dot_product = 0.0;
        for (size_t i = 0; i < length; i++) dot_product += a[i] * b[i];
        return dot_product;
    };

    void mean_normalize(float ** mat, int n_rows, int n_cols) {
        for (int i = 0; i < n_rows; i++) {
            float total = 0.0;
            for (int j = 0; j < n_cols; j++) {
                total += mat[i][j];
            }
            float mean = total / n_cols;
            for (int j = 0; j < n_cols; j++) {
                mat[i][j] -= mean;
            }
        }
    };

    float cosine_similarity(float a[], float b[], size_t length) {
        float numerator = dot_product(a, b, length);
        float a_norm = std::sqrt(dot_product(a, a, length));
        float b_norm = std::sqrt(dot_product(b, b, length));
        return numerator / (a_norm * b_norm);
    }

};