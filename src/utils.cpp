#include "utils.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm> // Add this for std::max_element
#include <iomanip>   // For formatted output

bool Utils::check_vocabs(const std::unordered_map<std::string, int>& vocab) {
    std::set<int> indices; // To track used indices
    std::set<std::string> strings; // To track duplicate strings
    int max_index = vocab.size() - 1;

    for (const auto& [word, index] : vocab) {
        // Check for duplicate strings
        if (strings.find(word) != strings.end()) {
            std::cerr << "Error: Duplicate word found in vocab: " << word << "\n";
            return false;
        }
        strings.insert(word);

        // Check if indices are within valid range and unique
        if (index < 0 || index > max_index) {
            std::cerr << "Error: Index " << index << " is out of range for word: " << word << "\n";
            return false;
        }
        if (indices.find(index) != indices.end()) {
            std::cerr << "Error: Duplicate index found in vocab: " << index << " for word: " << word << "\n";
            return false;
        }
        indices.insert(index);
    }

    // Check if all indices are contiguous from 0 to max_index
    if (indices.size() != vocab.size()) {
        std::cerr << "Error: Missing indices. Vocab size is " << vocab.size()
                  << " but unique indices are " << indices.size() << "\n";
        return false;
    }

    for (int i = 0; i <= max_index; ++i) {
        if (indices.find(i) == indices.end()) {
            std::cerr << "Error: Missing index " << i << " in vocab.\n";
            return false;
        }
    }

    std::cout << "Vocab validation passed: No duplicates, and indices are contiguous.\n";
    return true;
}

float Utils::leaky_relu(float x) {
    return (x > 0) ? x : GLOBAL_LEAKY_SLOPE * x;
}

std::vector<std::vector<float>> Utils::leaky_relu(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> output = input;
    for (auto& row : output) {
        for (auto& val : row) {
            val = leaky_relu(val);
        }
    }
    return output;
}


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

std::vector<std::vector<float>> Utils::mask_padding(
    const std::vector<std::vector<float>> &matrix,
    const std::vector<int> &padding_mask)
{
    std::vector<std::vector<float>> masked_matrix = matrix;
    for (size_t i = 0; i < masked_matrix.size(); ++i)
    {
        if (padding_mask[i] == 0)
        { // If it's a [PAD] token
            std::fill(masked_matrix[i].begin(), masked_matrix[i].end(), 0.0f);
        }
    }
    return masked_matrix;
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