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
        output = layer_norms[i*2].forward(add_matrices(residual, output));// Residual + Attentions

        // Mask padding in normalization output
        output = Utils::mask_padding(output, padding_mask);

        // Save the input for residual connection before FeedForward
        residual = output;

        // Apply FeedForward
        output = feed_forward_layers[i].forward(output);

        // Mask padding in feedforward output
        output = Utils::mask_padding(output, padding_mask);

        // Add residual connection and apply layer normalization
        output = layer_norms[i*2+1].forward(add_matrices(residual, output));// Residual + FFN

        // Mask padding in final normalized output
        output = Utils::mask_padding(output, padding_mask);
    }

    return output;
}

/*
// Backward pass implementation for the Transformer
void Transformer::backward(const std::vector<std::vector<float>>& grad_pooled) {
    // Step 1: Backpropagate through the final feedforward layer
    auto grad_ff = feed_forward_layers.back().backward(grad_pooled);

    // Step 2: Backpropagate through the attention layers in reverse order
    for (int i = attention_layers.size() - 1; i >= 0; --i) {
        // Step 2.1: Backpropagate through the feedforward layers within the block
        auto grad_ffn = feed_forward_layers[i].backward(grad_ff);

        // Step 2.2: Backpropagate through residual connections and layer normalization
        grad_ffn = layer_norms[i].backward(grad_ffn);

        // Step 2.3: Backpropagate through the multi-head attention layer
        auto grad_attn = attention_layers[i].backward(grad_ffn);

        // Update the gradient for the next block
        grad_ff = grad_attn;
    }

    // Step 3: Backpropagate through positional encoding
    auto grad_pos = pos_encoding.backward(grad_ff);

    // Step 4: Backpropagate through the embedding layer
    embedding.backward(grad_pos);
}

*/

