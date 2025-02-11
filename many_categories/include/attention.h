#ifndef ATTENTION_H
#define ATTENTION_H

#include <vector>
#include "config.h"
#include <iostream> // For std::cout and std::endl
#include <fstream>  // For file I/O

class MultiHeadAttention {
public:
    MultiHeadAttention(int d_model, int num_heads, bool load_parameters_yes_no, int layer_index);

    std::vector<std::vector<float>> forward(
        const std::vector<std::vector<float>>& query,
        const std::vector<std::vector<float>>& key,
        const std::vector<std::vector<float>>& value,
        const std::vector<int>& padding_mask,
        int head_cnt
    );

    // Backward pass to compute gradients
    std::vector<std::vector<float>> backward(
        const std::vector<std::vector<float>>& grad_output, // Gradient from the next layer
        int head_number
    );

    // New method to update weights AFTER backward pass
    void update_weights();

    // Save weights to binary files
    void save_weights(int layer_index);
    float read_weight(const std::string& matrix_type, int row, int col) const;


#ifdef TEST_ATTENTION
public:
#else
    #ifdef PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION
    public:
    #else
    private:
    #endif
#endif

    // (Optionally public for testing or debugging)
    std::vector<std::vector<float>> scaled_dot_product_attention(
        const std::vector<std::vector<float>>& query,
        const std::vector<std::vector<float>>& key,
        const std::vector<std::vector<float>>& value,
        const std::vector<int>& padding_mask
    );

    // ——————————————
    // Learned Weights
    // ——————————————
    std::vector<std::vector<float>> weights_q; // Query weights
    std::vector<std::vector<float>> weights_k; // Key weights
    std::vector<std::vector<float>> weights_v; // Value weights

//private:
public:
    // Static constants for storing/loading
    static const std::string file_prefix_attention_weights_q_layer_;
    static const std::string file_prefix_attention_weights_k_layer_;
    static const std::string file_prefix_attention_weights_v_layer_;

    // ——————————————
    // Momentum buffers
    // ——————————————
    std::vector<std::vector<float>> velocity_q; 
    std::vector<std::vector<float>> velocity_k;
    std::vector<std::vector<float>> velocity_v;

    // ——————————————
    // Gradients
    // ——————————————
    std::vector<std::vector<float>> grad_weights_q;
    std::vector<std::vector<float>> grad_weights_k;
    std::vector<std::vector<float>> grad_weights_v;

    std::vector<std::vector<float>> grad_query_full_output;

    // ——————————————————
    // Caches for backprop
    // ——————————————————
    std::vector<std::vector<float>> query_cache; 
    std::vector<std::vector<float>> key_cache;   
    std::vector<std::vector<float>> value_cache; 
    std::vector<std::vector<float>> attention_probs_cache; // Post-softmax attention distribution
    int num_heads;
    
    // ------ Caches of partial slices and results ------
    // For each head, store:
    //   - the "input slice" to the matmul (Q/K/V)
    //   - the "local weight" for Q/K/V
    //   - the "output" of the matmul (the Q_local etc. that go into attention)
    //   - the final attention output for that head

    // Q/K/V input slices: batch_size x segment_size
    std::vector<std::vector<std::vector<float>>> Q_local_input_cache;
    std::vector<std::vector<std::vector<float>>> K_local_input_cache;
    std::vector<std::vector<std::vector<float>>> V_local_input_cache;

    // local weight slices: segment_size x segment_size
    std::vector<std::vector<std::vector<float>>> W_q_local_cache;
    std::vector<std::vector<std::vector<float>>> W_k_local_cache;
    std::vector<std::vector<std::vector<float>>> W_v_local_cache;

    // outputs of the linear transforms (Q_local, K_local, V_local)
    std::vector<std::vector<std::vector<float>>> Q_local_cache;
    std::vector<std::vector<std::vector<float>>> K_local_cache;
    std::vector<std::vector<std::vector<float>>> V_local_cache;



};

#endif // ATTENTION_H
