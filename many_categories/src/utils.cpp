#include "utils.h"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm> // Add this for std::max_element
#include <iomanip>   // For formatted output

#include <vector>
#include <algorithm>
#include <cmath>
#include <random>

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
float Utils::leaky_relu_derivative(float x) {
    return (x > 0) ? 1.0f : GLOBAL_LEAKY_SLOPE;
}

std::vector<std::vector<float>> Utils::leaky_relu_derivative(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> output = input;
    for (auto& row : output) {
        for (auto& val : row) {
            val = leaky_relu_derivative(val);
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





/**
 * denoise_softmax()
 *
 * Applies uniform random noise to each logit based on the standard deviation
 * of the input logits, then applies a standard softmax.
 *
 * @param input                      The logits vector.
 * @param softmax_denoise_factor    Hyperparameter controlling the noise scale.
 * @return                           Softmax probabilities after noise injection.
 */
std::vector<float> Utils::denoise_softmax(const std::vector<float> &input, float softmax_denoise_factor = 0.05f)
{
    // Edge case: if input is empty or only one element, just return normal softmax
    if (input.size() <= 1) {
        return Utils::softmax(input);
    }
    
    // 1. Compute mean
    float sum = 0.0f;
    for (auto &val : input) {
        sum += val;
    }
    float mean = sum / static_cast<float>(input.size());

    // 2. Compute standard deviation
    float variance = 0.0f;
    for (auto &val : input) {
        float diff = val - mean;
        variance += diff * diff;
    }
    variance /= static_cast<float>(input.size());
    float stddev = std::sqrt(variance);

    // 3. Determine random range based on std dev
    float softmax_random_range = stddev * softmax_denoise_factor;

    // 4. Create a random engine & distribution
    //    For reproducibility, you might make this random engine a class member.
    //    For demonstration, we seed it with a random_device here.
    static std::random_device rd; 
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-softmax_random_range, softmax_random_range);

    // 5. Create a noisy version of the input
    std::vector<float> noisy_input(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        // Add uniform noise in [-softmax_random_range, +softmax_random_range]
        float noise = dist(gen);
        noisy_input[i] = input[i] + noise;
    }

    // 6. Return the standard softmax on the noisy input
    return Utils::softmax(noisy_input);
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
/**
 * @brief Computes gradient w.r.t. the input of a softmax function
 *        (row-wise softmax) for a 2D matrix.
 *
 * @param grad_output   Gradient from the next layer, same shape as softmax_out.
 *                      For each row i, grad_output[i] is dL/d(softmax_out[i]).
 * @param softmax_out   The output of softmax(input) row-by-row.
 *                      For each row i, softmax_out[i] sums to 1.
 * @return              Gradient w.r.t. the original 2D input of softmax, same shape.
 */

namespace Utils {
std::vector<std::vector<float>> softmax_backward(
    const std::vector<std::vector<float>>& grad_output,
    const std::vector<std::vector<float>>& softmax_out
) {
    // Prepare dInput with the same shape as grad_output.
    std::vector<std::vector<float>> dInput(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); ++i) {
        dInput[i].resize(grad_output[i].size(), 0.0f);
    }

    // Process each row independently
    for (size_t row = 0; row < grad_output.size(); ++row) {
        // 1) Compute dot = sum_j(grad_output[row][j] * softmax_out[row][j])
        float dot = 0.0f;
        for (size_t col = 0; col < grad_output[row].size(); ++col) {
            dot += grad_output[row][col] * softmax_out[row][col];
        }

        // 2) Compute dInput[row][col] = softmax_out[row][col] * (grad_output[row][col] - dot)
        for (size_t col = 0; col < grad_output[row].size(); ++col) {
            dInput[row][col] = softmax_out[row][col] * (grad_output[row][col] - dot);
        }
    }

    return dInput;
}
}
void Utils::scale_inplace(std::vector<std::vector<float>>& matrix, float scale_factor)
{
    for (auto& row : matrix) {
        for (auto& value : row) {
            value *= scale_factor;
        }
    }
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