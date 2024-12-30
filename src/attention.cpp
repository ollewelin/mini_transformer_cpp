#include "attention.h"
#include "utils.h"
#include <cmath>

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : weights_q(d_model, std::vector<float>(d_model, 0.0)),
      weights_k(d_model, std::vector<float>(d_model, 0.0)),
      weights_v(d_model, std::vector<float>(d_model, 0.0))
{
    // Initialize weights here if needed
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
    auto attention_output = scaled_dot_product_attention(Q, K, V);

    return attention_output;
}

std::vector<std::vector<float>> MultiHeadAttention::scaled_dot_product_attention(
    const std::vector<std::vector<float>> &query,
    const std::vector<std::vector<float>> &key,
    const std::vector<std::vector<float>> &value)
{
    // 1. Compute QK^T
    auto scores = Utils::matmul(query, Utils::transpose(key));

    // 2. Scale scores by sqrt(d_k)
    float scale_factor = std::sqrt(static_cast<float>(key[0].size()));
    for (size_t i = 0; i < scores.size(); ++i) {
        for (size_t j = 0; j < scores[0].size(); ++j) {
            scores[i][j] /= scale_factor;
        }
    }

    // 3. Apply softmax to scores
    for (size_t i = 0; i < scores.size(); ++i) {
        scores[i] = Utils::softmax(scores[i]);
    }

    // 4. Multiply scores with V
    auto output = Utils::matmul(scores, value);
    return output;
}
