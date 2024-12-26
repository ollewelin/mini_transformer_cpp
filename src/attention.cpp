#include "attention.h"

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
{
}

std::vector<std::vector<float>> MultiHeadAttention::forward(const std::vector<std::vector<float>> &query, const std::vector<std::vector<float>> &key, const std::vector<std::vector<float>> &value)
{
    return std::vector<std::vector<float>>();
}

std::vector<std::vector<float>> MultiHeadAttention::scaled_dot_product_attention(const std::vector<std::vector<float>> &query, const std::vector<std::vector<float>> &key, const std::vector<std::vector<float>> &value)
{
    return std::vector<std::vector<float>>();
}
