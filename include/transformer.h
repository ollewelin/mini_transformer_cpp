#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <numeric>
#include "embedding.h"
#include "positional_encoding.h"
#include "attention.h"
#include "feed_forward.h"
#include "config.h"

class Transformer
{
public:
    Transformer(int vocab_size, int d_model, int max_len, int num_heads, int d_ff, int num_layers, bool load_parameters_yes_no)
        : embedding(vocab_size, d_model, load_parameters_yes_no), // Initialize nested object here
          pos_encoding(max_len, d_model)           // Initialize another nested object here

    {
        num_layers_local = num_layers;
        for (int i = 0; i < num_layers; ++i)
        {
            attention_layers.emplace_back(d_model, num_heads, load_parameters_yes_no, i);
            feed_forward_layers.emplace_back(d_model, d_ff, load_parameters_yes_no, i);

            // Initialize gamma and beta for layer normalization
            gamma.emplace_back(d_model, 1.0f); // Default scaling: 1.0
            beta.emplace_back(d_model, 0.0f);  // Default shifting: 0.0

            if (load_parameters_yes_no)
            {
                // Load gamma and beta from file
                std::string file_name = "normalize_weights_layer_" + std::to_string(i) + ".bin";
                std::ifstream file(file_name, std::ios::binary);
                if (file.is_open())
                {
                    file.read(reinterpret_cast<char *>(gamma[i].data()), gamma[i].size() * sizeof(float));
                    file.read(reinterpret_cast<char *>(beta[i].data()), beta[i].size() * sizeof(float));
                    file.close();
                }
                else
                {
                    std::cerr << "Warning: Could not load normalization weights for layer " << i << ". Using defaults.\n";
                }
            }
        }

        std::cout << "Transformer initialized with " << num_layers << " layers." << std::endl;
    };
    std::vector<std::vector<float>> forward(const std::vector<int> &input);//Overall tranformer operation function
    void save_layer_norm_weights();// Saving Layer Normalization weights
    void save_embedding_matrix();
    void save_attention_weights();
    void save_feed_forward_weights();    
private:
    Embedding embedding;
    PositionalEncoding pos_encoding;
    std::vector<MultiHeadAttention> attention_layers;
    std::vector<FeedForward> feed_forward_layers;

    // Store normalization parameters for each layer
    std::vector<std::vector<float>> gamma; // Scaling parameters for each layer
    std::vector<std::vector<float>> beta;  // Shifting parameters for each layer

    // Helper functions
    std::vector<std::vector<float>> add_matrices(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b);
    std::vector<std::vector<float>> layer_normalize(const std::vector<std::vector<float>> &input, size_t layer_index);
    int num_layers_local;

};


#endif // TRANSFORMER_H


