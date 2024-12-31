#include "transformer.h"



std::vector<std::vector<float>> Transformer::forward(const std::vector<int> &input) {
    // Embedding and positional encoding will be applied here.
    std::vector<std::vector<float>> output = embedding.forward(input);
    output = pos_encoding.add_positional_encoding(output);

    // Apply each layer's MultiHeadAttention
    for (auto &attention_layer : attention_layers) {
        output = attention_layer.forward(output, output, output); // Self-attention
    }

    return output;
}
