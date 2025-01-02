#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include "config.h"
#include <iostream>

namespace Utils {
    // Matrix multiplication
    std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
    
    // Transpose a matrix
    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix);
    
    // Apply softmax to a vector
    std::vector<float> softmax(const std::vector<float>& input);
    
    // Print a matrix
    void print_matrix(const std::vector<std::vector<float>>& matrix);
    
    // Check matrix dimensions
    void check_matrices(const std::vector<std::vector<float>>& A,
                        const std::vector<std::vector<float>>& B,
                        const std::vector<std::vector<float>>& C);

    // Leaky ReLU for a single float
    float leaky_relu(float x);

    // Leaky ReLU for a matrix
    std::vector<std::vector<float>> leaky_relu(const std::vector<std::vector<float>>& input);
};

#endif // UTILS_H


