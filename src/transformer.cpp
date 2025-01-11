#include "transformer.h"
using namespace std;
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

void Transformer::save_LayerNormalization_weights()
{
    for (int i = 0; i < num_layers_local; i++)
    {
        //TODO
       // layer_norms[i*2].save_parameters(i*2);
       // layer_norms[i*2+1].save_parameters(i*2+1);
    }
}
std::vector<std::vector<float>> Transformer::forward(const std::vector<int>& input, const std::vector<int>& padding_mask) {
    // Step 1: Embedding and positional encoding
    std::vector<std::vector<float>> output = embedding.forward(input);
    output = pos_encoding.add_positional_encoding(output);
    input_tokens = input;//Used for backprop
    // Clear caches for this pass

    residual_connections.clear();
    attention_outputs.clear();
    normalized_attention_outputs.clear();
    feedforward_outputs.clear();
    normalized_feedforward_outputs.clear();

    // Step 2: Iterate through attention and feedforward layers
    for (size_t i = 0; i < attention_layers.size(); ++i) {
        // Save the input for residual connection
        residual_connections.push_back(output);

        // Apply MultiHeadAttention with padding mask
        auto attention_output = attention_layers[i].forward(output, output, output, padding_mask);
    //    attention_outputs.push_back(attention_output);

        // Mask padding in attention output
        auto masked_attention_output = Utils::mask_padding(attention_output, padding_mask);

        // Add residual connection and apply layer normalization
       // auto norm_attention_output = layer_norms[i * 2].forward(add_matrices(residual_connections.back(), masked_attention_output));
        auto norm_attention_output = add_matrices(residual_connections.back(), masked_attention_output);
    //    normalized_attention_outputs.push_back(norm_attention_output);

        // Mask padding in normalization output
        output = Utils::mask_padding(norm_attention_output, padding_mask);

        // Save the input for residual connection before FeedForward
        residual_connections.push_back(output);

        // Apply FeedForward
        auto feedforward_output = feed_forward_layers[i].forward(output);
      //  feedforward_outputs.push_back(feedforward_output);

        // Mask padding in feedforward output
        auto masked_feedforward_output = Utils::mask_padding(feedforward_output, padding_mask);

        // Add residual connection and apply layer normalization
       // auto norm_feedforward_output = layer_norms[i * 2 + 1].forward(add_matrices(residual_connections.back(), masked_feedforward_output));
        auto norm_feedforward_output = add_matrices(residual_connections.back(), masked_feedforward_output);
    //    normalized_feedforward_outputs.push_back(norm_feedforward_output);

        // Mask padding in final normalized output
        output = Utils::mask_padding(norm_feedforward_output, padding_mask);
    }

    return output;
}

/*
std::vector<std::vector<float>> Transformer::backward(const std::vector<std::vector<float>>& grad_pooled) {
    auto grad_ff = feed_forward_layers.back().backward(grad_pooled);

    for (int i = attention_layers.size() - 1; i >= 0; --i) {
        // Backpropagate through feedforward layers
        auto grad_ffn = feed_forward_layers[i].backward(grad_ff);

        // Backpropagate through residual connections and layer normalization
     //   grad_ffn = layer_norms[i * 2 + 1].backward(grad_ffn, feedforward_outputs[i]);

        // Backpropagate through attention layers
        auto grad_attn = attention_layers[i].backward(grad_ffn);

        // Backpropagate through residual connections and layer normalization
     //   grad_ff = layer_norms[i * 2].backward(grad_attn, attention_outputs[i]);
    }

    // Backpropagate through positional encoding
    auto grad_pos = pos_encoding.backward(grad_ff);

    // Backpropagate through embedding
    // Assuming `input_tokens` is the tokenized input to the Transformer (from the forward pass)
    float learning_rate = 0.01;
    embedding.apply_gradients(input_tokens, grad_pos, learning_rate);
    
    return grad_pos; // Return gradient for external validation (optional)
}
*/


std::vector<std::vector<float>> Transformer::backward(const std::vector<std::vector<float>>& grad_pooled) {
    auto grad_ff = feed_forward_layers.back().backward(grad_pooled);
    residual_connections.clear();
    for (int i = attention_layers.size() - 1; i >= 0; --i) {
        residual_connections.clear();
        residual_connections.push_back(grad_ff);
        // Backprop feedforward
        auto grad_ffn = feed_forward_layers[i].backward(grad_ff);
        grad_ffn = add_matrices(grad_ffn, residual_connections.back());
        // --- NEW: Update feed-forward weights after backward
        feed_forward_layers[i].update_weights();
        // Backprop attention
        residual_connections.clear();
        residual_connections.push_back(grad_ff);
        auto grad_attn = attention_layers[i].backward(grad_ffn);
        attention_layers[i].update_weights();
        grad_ffn = add_matrices(grad_ffn, residual_connections.back());
        
        // --- You would similarly call attention_layers[i].update_weights(), 
        //     if you implement a similar update method for the attention layer.
    }

    // Backprop positional encoding
    auto grad_pos = pos_encoding.backward(grad_ff);
    
    // Backprop embedding
    embedding.apply_gradients(input_tokens, grad_pos, GLOBAL_learning_rate);

    return grad_pos;
}
