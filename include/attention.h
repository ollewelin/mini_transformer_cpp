#ifndef ATTENTION_H
#define ATTENTION_H
#include <vector>
#include "config.h"

class MultiHeadAttention {
public:
    MultiHeadAttention(int d_model, int num_heads, bool load_parameters_yes_no, int layer_index);
    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& query,
        const std::vector<std::vector<float>>& key,
        const std::vector<std::vector<float>>& value);

private:
#ifdef TEST_ATTENTION
public:    
#endif
    std::vector<std::vector<float>> scaled_dot_product_attention(
        const std::vector<std::vector<float>>& query,
        const std::vector<std::vector<float>>& key,
        const std::vector<std::vector<float>>& value);

    std::vector<std::vector<float>> weights_q; // Query weights
    std::vector<std::vector<float>> weights_k; // Key weights
    std::vector<std::vector<float>> weights_v; // Value weights
};

#endif // ATTENTION_H
