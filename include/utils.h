#include <vector>
#include "config.h"
#include <iostream>
namespace Utils {
    std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix);
    std::vector<float> softmax(const std::vector<float>& input);
        // Function to print matrices
    void print_matrix(const std::vector<std::vector<float>>& matrix);
    // Function to check matrix dimensions
    void check_matrices(const std::vector<std::vector<float>>& A,
                        const std::vector<std::vector<float>>& B,
                        const std::vector<std::vector<float>>& C);    
};
