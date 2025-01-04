#include "transformer.h"

std::vector<std::vector<float>> Transformer::add_matrices(const std::vector<std::vector<float>> &a, const std::vector<std::vector<float>> &b) {
    if (a.size() != b.size() || a[0].size() != b[0].size()) {
        throw std::invalid_argument("Matrix dimensions must match for addition.");
    }
    std::vector<std::vector<float>> result = a;
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < a[0].size(); ++j) {
            result[i][j] += b[i][j];
        }
    }
    return result;
}
void Transformer::save_embedding_matrix()
{
    embedding.save_embedding_matrix();
}
void Transformer::save_attention_weights()
{
    for(int i=0;i<num_layers_local;i++)
    {
        attention_layers[i].save_weights(i);
    }
}
void Transformer::save_feed_forward_weights()
{
    for (int i = 0; i < num_layers_local; i++)
    {
        feed_forward_layers[i].save_weights(i);
    }
}

std::vector<std::vector<float>> Transformer::layer_normalize(const std::vector<std::vector<float>> &input, size_t layer_index) {
    std::vector<std::vector<float>> output = input;

    // Get gamma and beta for this layer
    const std::vector<float> &gamma_layer = gamma[layer_index];
    const std::vector<float> &beta_layer = beta[layer_index];

    for (size_t i = 0; i < input.size(); ++i) { // For each row
        const auto &row = input[i];
        float mean = std::accumulate(row.begin(), row.end(), 0.0f) / row.size();
        float variance = 0.0f;

        for (float val : row) {
            variance += (val - mean) * (val - mean);
        }
        variance /= row.size();
        float stddev = std::sqrt(variance + 1e-6); // Epsilon for numerical stability

        // Normalize and apply gamma and beta
        for (size_t j = 0; j < row.size(); ++j) {
            output[i][j] = gamma_layer[j] * ((row[j] - mean) / stddev) + beta_layer[j];
        }
    }

    return output;
}

void Transformer::save_layer_norm_weights()
{
    for (size_t i = 0; i < gamma.size(); ++i)
    {
        std::string file_name = "normalize_weights_layer_" + std::to_string(i) + ".bin";
        std::ofstream file(file_name, std::ios::binary);
        if (file.is_open())
        {
            file.write(reinterpret_cast<const char *>(gamma[i].data()), gamma[i].size() * sizeof(float));
            file.write(reinterpret_cast<const char *>(beta[i].data()), beta[i].size() * sizeof(float));
            file.close();
        }
        else
        {
            std::cerr << "Error: Could not save normalization weights for layer " << i << ".\n";
        }
    }
}

std::vector<std::vector<float>> Transformer::forward(const std::vector<int>& input, const std::vector<int>& padding_mask) {
    // Step 1: Embedding and positional encoding
    std::vector<std::vector<float>> output = embedding.forward(input);
    output = pos_encoding.add_positional_encoding(output);

    // Step 2: Iterate through attention and feedforward layers
    for (size_t i = 0; i < attention_layers.size(); ++i) {
        // Save the input for residual connection
        auto residual = output;

        // Apply MultiHeadAttention with padding mask
        output = attention_layers[i].forward(output, output, output, padding_mask);// Self-attention

        // Mask padding in attention output
        output = Utils::mask_padding(output, padding_mask);

        // Add residual connection and apply layer normalization
        output = layer_normalize(add_matrices(residual, output), i);// Residual + Attentions

        // Mask padding in normalization output
        output = Utils::mask_padding(output, padding_mask);

        // Save the input for residual connection before FeedForward
        residual = output;

        // Apply FeedForward
        output = feed_forward_layers[i].forward(output);

        // Mask padding in feedforward output
        output = Utils::mask_padding(output, padding_mask);

        // Add residual connection and apply layer normalization
        output = layer_normalize(add_matrices(residual, output), i);// Residual + FFN

        // Mask padding in final normalized output
        output = Utils::mask_padding(output, padding_mask);
    }

    return output;
}




