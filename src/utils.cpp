#include "utils.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm> // Add this for std::max_element
#include <iomanip>   // For formatted output

namespace Utils {

    // Function to check matrix dimensions
    void check_matrices(const std::vector<std::vector<float>>& A,
                        const std::vector<std::vector<float>>& B,
                        const std::vector<std::vector<float>>& C) {
        if (A.size() != B.size() || A.size() != C.size()) {
            std::cout << "Error: Number of rows in Q, K, and V must match. Program terminated.\n";
            exit(1);
        }
        for (size_t i = 0; i < A.size(); ++i) {
            if (A[i].size() != B[i].size() || A[i].size() != C[i].size()) {
                std::cout << "Error: Row " << i << " of Q, K, and V must have the same number of columns. Program terminated.\n";
                exit(1);
            }
        }
        std::cout << "Matrix sizes are valid.\n";
    }

    // Existing functions (matmul, transpose, softmax, etc.) remain here
}


std::vector<std::vector<float>> Utils::matmul(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b)
{
    if (a[0].size() != b.size()) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    size_t rows = a.size();
    size_t cols = b[0].size();
    size_t inner_dim = b.size();

    std::vector<std::vector<float>> result(rows, std::vector<float>(cols, 0.0f));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            for (size_t k = 0; k < inner_dim; ++k) {
                result[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return result;
}


std::vector<std::vector<float>> Utils::transpose(const std::vector<std::vector<float>> &matrix)
{
    size_t rows = matrix.size();
    size_t cols = matrix[0].size();

    std::vector<std::vector<float>> result(cols, std::vector<float>(rows));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

std::vector<float> Utils::softmax(const std::vector<float> &input)
{
    float max_input = *std::max_element(input.begin(), input.end()); // For numerical stability
    float sum_exp = 0.0f;

    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = std::exp(input[i] - max_input);
        sum_exp += result[i];
    }

    for (size_t i = 0; i < input.size(); ++i) {
        result[i] /= sum_exp;
    }

    return result;
}

void Utils::print_matrix(const std::vector<std::vector<float>>& matrix)
{
    std::cout << "[\n";
    for (const auto& row : matrix) {
        std::cout << "  [";
        for (size_t i = 0; i < row.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << row[i];
            if (i < row.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << "]\n";
    }
    std::cout << "]\n";
}