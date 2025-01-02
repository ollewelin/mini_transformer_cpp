#include "attention.h"
#include "utils.h"
#include <cmath>
#include <cstdlib>  // For std::rand
#include <ctime>    // For std::time
#include <iostream> // For std::cout and std::endl
#include <fstream>  // For file I/O

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, bool load_parameters_yes_no, int layer_index)
    : weights_q(d_model, std::vector<float>(d_model, 0.0)),
      weights_k(d_model, std::vector<float>(d_model, 0.0)),
      weights_v(d_model, std::vector<float>(d_model, 0.0))
{
    const std::string weights_q_file = "attention_weights_q_layer_" + std::to_string(layer_index) + ".bin";
    const std::string weights_k_file = "attention_weights_k_layer_" + std::to_string(layer_index) + ".bin";
    const std::string weights_v_file = "attention_weights_v_layer_" + std::to_string(layer_index) + ".bin";

    bool loaded = false;

    if (load_parameters_yes_no) {
        std::ifstream file_q(weights_q_file, std::ios::binary);
        std::ifstream file_k(weights_k_file, std::ios::binary);
        std::ifstream file_v(weights_v_file, std::ios::binary);

        if (file_q.is_open() && file_k.is_open() && file_v.is_open()) {
            for (auto& row : weights_q) {
                file_q.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            for (auto& row : weights_k) {
                file_k.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            for (auto& row : weights_v) {
                file_v.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            file_q.close();
            file_k.close();
            file_v.close();

            std::cout << "Attention weights for layer " << layer_index << " loaded from files." << std::endl;
            loaded = true; // Mark as successfully loaded
        } else {
            std::cerr << "Warning: Could not open weight files for layer " << layer_index << ". Falling back to random initialization." << std::endl;
        }
    }

    if (!loaded) {
        std::srand(std::time(0));
        float scale = std::sqrt(2.0f / d_model);

        for (auto& row : weights_q) {
            for (auto& val : row) {
                val = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; // Uniform [-scale, scale]
            }
        }

        for (auto& row : weights_k) {
            for (auto& val : row) {
                val = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; // Uniform [-scale, scale]
            }
        }

        for (auto& row : weights_v) {
            for (auto& val : row) {
                val = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; // Uniform [-scale, scale]
            }
        }

        std::cout << "Attention weights for layer " << layer_index << " initialized with random values." << std::endl;

        // Save the randomized weights to binary files
        std::ofstream save_file_q(weights_q_file, std::ios::binary);
        std::ofstream save_file_k(weights_k_file, std::ios::binary);
        std::ofstream save_file_v(weights_v_file, std::ios::binary);

        if (save_file_q.is_open() && save_file_k.is_open() && save_file_v.is_open()) {
            for (const auto& row : weights_q) {
                save_file_q.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }
            for (const auto& row : weights_k) {
                save_file_k.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }
            for (const auto& row : weights_v) {
                save_file_v.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
            }
            save_file_q.close();
            save_file_k.close();
            save_file_v.close();

            std::cout << "Randomized attention weights for layer " << layer_index << " saved to files." << std::endl;
        } else {
            std::cerr << "Error: Could not save attention weights for layer " << layer_index << " to files." << std::endl;
            exit(EXIT_FAILURE);
        }
    }
#ifdef PRINT_OUT_INIT_VECTORS
    // Print a few rows of weights_q, weights_k, and weights_v
    std::cout << "\nSample rows of weights_q for layer " << layer_index << ":" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, weights_q.size()); ++i) {
        for (float val : weights_q[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nSample rows of weights_k for layer " << layer_index << ":" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, weights_k.size()); ++i) {
        for (float val : weights_k[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nSample rows of weights_v for layer " << layer_index << ":" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, weights_v.size()); ++i) {
        for (float val : weights_v[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
#endif
}

std::vector<std::vector<float>> MultiHeadAttention::forward(
    const std::vector<std::vector<float>> &query,
    const std::vector<std::vector<float>> &key,
    const std::vector<std::vector<float>> &value)
{
    // 1. Linear transformations for Q, K, V
    auto Q = Utils::matmul(query, weights_q); // Query * weights_q
    auto K = Utils::matmul(key, weights_k);   // Key * weights_k
    auto V = Utils::matmul(value, weights_v); // Value * weights_v

    // 2. Scaled dot-product attention
#ifndef PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION     
    auto attention_output = scaled_dot_product_attention(Q, K, V);
#else
    auto attention_output = scaled_dot_product_attention_with_printout(Q, K, V);
#endif

    return attention_output;
}


#ifndef PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION  
std::vector<std::vector<float>> MultiHeadAttention::scaled_dot_product_attention(
    const std::vector<std::vector<float>> &query,
    const std::vector<std::vector<float>> &key,
    const std::vector<std::vector<float>> &value)
{
    // 1. Compute QK^T
    auto scores = Utils::matmul(query, Utils::transpose(key));

    // 2. Scale scores by sqrt(d_k)
    float scale_factor = std::sqrt(static_cast<float>(key[0].size()));
    for (size_t i = 0; i < scores.size(); ++i)
    {
        for (size_t j = 0; j < scores[0].size(); ++j)
        {
            scores[i][j] /= scale_factor;
        }
    }

    // 3. Apply masking to prevent attending to future tokens
    for (size_t i = 0; i < scores.size(); ++i)
    {
        for (size_t j = 0; j < scores[i].size(); ++j)
        {
            if (j > i) // Mask future positions
            {
                scores[i][j] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // 4. Apply softmax to scores
    for (size_t i = 0; i < scores.size(); ++i)
    {
        scores[i] = Utils::softmax(scores[i]);
    }

    // 5. Multiply scores with V
    auto output = Utils::matmul(scores, value);
    return output;
}

#else // Use of PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION in config.h
std::vector<std::vector<float>> MultiHeadAttention::scaled_dot_product_attention_with_printout(
    const std::vector<std::vector<float>> &query,
    const std::vector<std::vector<float>> &key,
    const std::vector<std::vector<float>> &value)
{
    using namespace std;

    cout << "\n=== Scaled Dot-Product Attention Debug Output ===\n";

    // 1. Compute QK^T
    cout << "\nStep 1: Compute QK^T\n";
    cout << "Query (Q) matrix (shape: " << query.size() << " x " << query[0].size() << "):\n";
    Utils::print_matrix(query); // Assuming Utils has a method to print matrices
    cout << "Key (K) matrix (shape: " << key.size() << " x " << key[0].size() << "):\n";
    Utils::print_matrix(key);

    auto scores = Utils::matmul(query, Utils::transpose(key));
    cout << "QK^T (scores matrix, shape: " << scores.size() << " x " << scores[0].size() << "):\n";
    Utils::print_matrix(scores);
    cout << "Each row in this matrix corresponds to a token's query, "
         << "and each column represents its dot product similarity with other tokens' keys.\n";

    // 2. Scale scores by sqrt(d_k)
    cout << "\nStep 2: Scale scores by sqrt(d_k)\n";
    float scale_factor = std::sqrt(static_cast<float>(key[0].size()));
    cout << "Scaling factor (sqrt(d_k)): " << scale_factor << endl;
    for (size_t i = 0; i < scores.size(); ++i)
    {
        for (size_t j = 0; j < scores[0].size(); ++j)
        {
            scores[i][j] /= scale_factor;
        }
    }
    cout << "Scaled scores matrix:\n";
    Utils::print_matrix(scores);
    cout << "Each score is scaled to adjust for the dimensionality of the key vectors.\n";
    
    // 3. Apply masking to prevent attending to future tokens
    cout << "\nStep 3: Apply masking to prevent attending to future tokens\n";
    for (size_t i = 0; i < scores.size(); ++i)
    {
        for (size_t j = 0; j < scores[i].size(); ++j)
        {
            if (j > i) // Mask future positions
            {
                scores[i][j] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    cout << "Masked scores matrix:\n";
    Utils::print_matrix(scores);
    cout << "This matrix shows the scores after applying a mask to ensure that a token only attends to itself and earlier tokens.\n";

    // 4. Apply softmax to scores
    cout << "\nStep 4: Apply softmax to scores\n";
    for (size_t i = 0; i < scores.size(); ++i)
    {
        scores[i] = Utils::softmax(scores[i]);
    }
    cout << "Softmax applied (attention weights):\n";
    Utils::print_matrix(scores);
    cout << "Each row represents the attention distribution for a token. "
         << "The values sum to 1, showing how much each token attends to other tokens.\n";

    // 5. Multiply scores with V
    cout << "\nStep 5: Multiply scores with Value (V) matrix\n";
    cout << "Value (V) matrix (shape: " << value.size() << " x " << value[0].size() << "):\n";
    Utils::print_matrix(value);

    auto output = Utils::matmul(scores, value);
    cout << "Output matrix (shape: " << output.size() << " x " << output[0].size() << "):\n";
    Utils::print_matrix(output);
    cout << "Each row in the output matrix corresponds to the weighted sum of value vectors "
         << "for each token, based on its attention distribution.\n";


    cout << "=== End of Debug Output ===\n";
    return output;
}
#endif
