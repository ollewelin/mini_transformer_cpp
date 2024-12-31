#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include "embedding.h"
#include "positional_encoding.h"
#include "attention.h"
#include "feed_forward.h"
#include <iostream>
class Transformer
{
public:
    Transformer(int vocab_size, int d_model, int max_len, int num_heads, int d_ff, int num_layers, bool load_parameters_yes_no)
        : embedding(vocab_size, d_model, load_parameters_yes_no), // Initialize nested object here
          pos_encoding(max_len, d_model)           // Initialize another nested object here
    {
        // Constructor body (if needed)
        for (int i = 0; i < num_layers; ++i)
        {
            attention_layers.emplace_back(d_model, num_heads, load_parameters_yes_no, i);
        }
        std::cout << "Transformer initialized with " << num_layers << " layers." << std::endl;
    };


    
    std::vector<std::vector<float>> forward(const std::vector<int> &input);

private:
    Embedding embedding;
    PositionalEncoding pos_encoding;
    std::vector<MultiHeadAttention> attention_layers;
    std::vector<FeedForward> feed_forward_layers;
};
#endif // TRANSFORMER_H

