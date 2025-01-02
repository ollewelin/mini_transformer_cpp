#ifndef EMBEDDING_H
#define EMBEDDING_H

#include <vector>
#include "config.h"

class Embedding {
public:
    // Constructor with optional file loading
    Embedding(int vocab_size, int d_model, bool load_parameters_yes_no);

    // Forward pass
    std::vector<std::vector<float>> forward(const std::vector<int>& input);

    // Apply gradients to the embedding matrix
    void apply_gradients(const std::vector<int>& input, const std::vector<std::vector<float>>& grad_embedding, float learning_rate);

private:
    std::vector<std::vector<float>> embedding_matrix; // The embedding lookup table
};

#endif // EMBEDDING_H
