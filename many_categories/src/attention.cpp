#include "attention.h"
#include "utils.h"
#include <cmath>
#include <cstdlib>  // For std::rand
#include <ctime>    // For std::time
#include <iostream> // For std::cout and std::endl
#include <fstream>  // For file I/O

const std::string MultiHeadAttention::file_prefix_attention_weights_q_layer_ = "attention_weights_q_layer_";
const std::string MultiHeadAttention::file_prefix_attention_weights_k_layer_ = "attention_weights_k_layer_";
const std::string MultiHeadAttention::file_prefix_attention_weights_v_layer_ = "attention_weights_v_layer_";

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads, bool load_parameters_yes_no, int layer_index)
    : 
      weights_q(d_model, std::vector<float>(d_model, 0.0f)),
      weights_k(d_model, std::vector<float>(d_model, 0.0f)),
      weights_v(d_model, std::vector<float>(d_model, 0.0f)),

      // Velocity (momentum) buffers
      velocity_q(d_model, std::vector<float>(d_model, 0.0f)),
      velocity_k(d_model, std::vector<float>(d_model, 0.0f)),
      velocity_v(d_model, std::vector<float>(d_model, 0.0f)),

      // Gradients
      grad_weights_q(d_model, std::vector<float>(d_model, 0.0f)),
      grad_weights_k(d_model, std::vector<float>(d_model, 0.0f)),
      grad_weights_v(d_model, std::vector<float>(d_model, 0.0f)),

      // grad_query_full_output
      grad_query_full_output(d_model, std::vector<float>(d_model, 0.0f)),

      // Caches
      query_cache(),
      key_cache(),
      value_cache(),
      num_heads(num_heads)
            

{
    const std::string weights_q_file = file_prefix_attention_weights_q_layer_ + std::to_string(layer_index) + ".bin";
    const std::string weights_k_file = file_prefix_attention_weights_k_layer_ + std::to_string(layer_index) + ".bin";
    const std::string weights_v_file = file_prefix_attention_weights_v_layer_ + std::to_string(layer_index) + ".bin";

    bool loaded = false;

    if (load_parameters_yes_no) {
        std::ifstream file_q(weights_q_file, std::ios::binary);
        std::ifstream file_k(weights_k_file, std::ios::binary);
        std::ifstream file_v(weights_v_file, std::ios::binary);

        if (file_q.is_open() && file_k.is_open() && file_v.is_open()) {
            for (auto& row : weights_q) {
                file_q.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            for (auto& row : weights_k) {
                file_k.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            for (auto& row : weights_v) {
                file_v.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            file_q.close();
            file_k.close();
            file_v.close();

            std::cout << "Attention weights for layer " << layer_index << " loaded from files." << std::endl;
            loaded = true; // Mark as successfully loaded
        } else {
            std::cerr << "Warning: Could not open weight files for layer " << layer_index << ". Falling back to random initialization." << std::endl;
        }
    }

    if (!loaded) {
        std::srand(std::time(0));
        float scale = std::sqrt(2.0f / d_model);

        for (auto& row : weights_q) {
            for (auto& val : row) {
                val = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; // Uniform [-scale, scale]
            }
        }

        for (auto& row : weights_k) {
            for (auto& val : row) {
                val = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; // Uniform [-scale, scale]
            }
        }

        for (auto& row : weights_v) {
            for (auto& val : row) {
                val = scale * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f; // Uniform [-scale, scale]
            }
        }

        std::cout << "Attention weights for layer " << layer_index << " initialized with random values." << std::endl;

    }

}


std::vector<std::vector<float>> MultiHeadAttention::backward(
    const std::vector<std::vector<float>>& grad_output,
    int head_number
) {
   
    //=============================================================
    // 0) Setup / Initialize local grad buffers
    //=============================================================
    size_t segment_size = query_cache[0].size() / num_heads;

    // Split input and cached tensors into smaller segments for this head
    std::vector<std::vector<float>> query_local_single_head(query_cache.size(), std::vector<float>(segment_size));
    std::vector<std::vector<float>> key_local_single_head(key_cache.size(), std::vector<float>(segment_size));
    std::vector<std::vector<float>> value_local_single_head(value_cache.size(), std::vector<float>(segment_size));

    std::vector<std::vector<float>> weights_q_local_single_head(segment_size, std::vector<float>(segment_size));
    std::vector<std::vector<float>> weights_k_local_single_head(segment_size, std::vector<float>(segment_size));
    std::vector<std::vector<float>> weights_v_local_single_head(segment_size, std::vector<float>(segment_size));

    // Split gradients for local computations
    std::vector<std::vector<float>> grad_output_local(grad_output.size(), std::vector<float>(segment_size));

    for (size_t i = 0; i < query_cache.size(); ++i) {
        for (size_t j = 0; j < segment_size; ++j) {
            int index = head_number * segment_size + j;
            query_local_single_head[i][j] = query_cache[i][index];
            key_local_single_head[i][j] = key_cache[i][index];
            value_local_single_head[i][j] = value_cache[i][index];
            grad_output_local[i][j] = grad_output[i][index];
        }
    }

    for (size_t i = 0; i < segment_size; ++i) {
        for (size_t j = 0; j < segment_size; ++j) {
            int index_col = head_number * segment_size + j;
            int index_row = head_number * segment_size + i;
            weights_q_local_single_head[i][j] = weights_q[index_row][index_col];
            weights_k_local_single_head[i][j] = weights_k[index_row][index_col];
            weights_v_local_single_head[i][j] = weights_v[index_row][index_col];
        }
    }

    //=============================================================
    // 1) Backprop through final matmul: output = A * V
    //=============================================================
    auto A = attention_probs_cache; // Cached attention probabilities (full size)

    auto dV = Utils::matmul(Utils::transpose(A), grad_output_local);

    auto V = Utils::matmul(value_local_single_head, weights_v_local_single_head);
    auto dA = Utils::matmul(grad_output_local, Utils::transpose(V));

    //=============================================================
    // 2) Backprop through softmax
    //=============================================================
    std::vector<std::vector<float>> dScores = Utils::softmax_backward(dA, A);

    //=============================================================
    // 3) Backprop through scaled dot-product attention
    //=============================================================
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(key_local_single_head[0].size()));

    auto Q = Utils::matmul(query_local_single_head, weights_q_local_single_head);
    auto K = Utils::matmul(key_local_single_head, weights_k_local_single_head);

    auto dQ = Utils::matmul(dScores, K);
    Utils::scale_inplace(dQ, scale_factor);

    auto dK = Utils::matmul(Utils::transpose(dScores), Q);
    Utils::scale_inplace(dK, scale_factor);

    //=============================================================
    // 4) Backprop through the linear transformations
    //=============================================================
    auto grad_weights_q_local_single_head = Utils::matmul(Utils::transpose(query_local_single_head), dQ);
    auto grad_weights_k_local_single_head = Utils::matmul(Utils::transpose(key_local_single_head), dK);
    auto grad_weights_v_local_single_head = Utils::matmul(Utils::transpose(value_local_single_head), dV);

    auto grad_query_local_single_head = Utils::matmul(dQ, Utils::transpose(weights_q_local_single_head));
    auto grad_key_local_single_head = Utils::matmul(dK, Utils::transpose(weights_k_local_single_head));
    auto grad_value_local_single_head = Utils::matmul(dV, Utils::transpose(weights_v_local_single_head));

    // plug in grad_weights_q_local_single_head to grad_weights_q
    // plug in grad_weights_k_local_single_head to grad_weights_k
    // plug in grad_weights_v_local_single_head to grad_weights_v

    // plug in grad_query_local_single_head into grad_query_full_output
    // not used auto grad_key_local_single_head
    // not used auto grad_value_local_single_head

    // Plug in `grad_weights_q_local_single_head` to `grad_weights_q`
    for (size_t i = 0; i < segment_size; ++i) {
        for (size_t j = 0; j < segment_size; ++j) {
            size_t index_row = head_number * segment_size + i;
            size_t index_col = head_number * segment_size + j;
            grad_weights_q[index_row][index_col] += grad_weights_q_local_single_head[i][j];
            grad_weights_k[index_row][index_col] += grad_weights_k_local_single_head[i][j];
            grad_weights_v[index_row][index_col] += grad_weights_v_local_single_head[i][j];
        }
    }

    // Plug in `grad_query_local_single_head` to `grad_query_full_output`
    for (size_t i = 0; i < grad_query_local_single_head.size(); ++i) {
        for (size_t j = 0; j < segment_size; ++j) {
            size_t index = head_number * segment_size + j;
            grad_query_full_output[i][index] += grad_query_local_single_head[i][j];
        }
    }

    //=============================================================
    // 5) Combine or return whichever gradient is relevant
    //=============================================================

   // return grad_query_full_output;
   return grad_output;
}




void MultiHeadAttention::save_weights(int layer_index) {
    const std::string weights_q_file = file_prefix_attention_weights_q_layer_ + std::to_string(layer_index) + ".bin";
    const std::string weights_k_file = file_prefix_attention_weights_k_layer_ + std::to_string(layer_index) + ".bin";
    const std::string weights_v_file = file_prefix_attention_weights_v_layer_ + std::to_string(layer_index) + ".bin";
    
    std::ofstream save_file_q(weights_q_file, std::ios::binary);
    std::ofstream save_file_k(weights_k_file, std::ios::binary);
    std::ofstream save_file_v(weights_v_file, std::ios::binary);

    if (save_file_q.is_open() && save_file_k.is_open() && save_file_v.is_open()) {
        for (const auto& row : weights_q) {
            save_file_q.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
        for (const auto& row : weights_k) {
            save_file_k.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
        for (const auto& row : weights_v) {
            save_file_v.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
        save_file_q.close();
        save_file_k.close();
        save_file_v.close();

        std::cout << "Attention weights for layer " << layer_index << " saved to files." << std::endl;
    } else {
        std::cerr << "Error: Could not save attention weights for layer " << layer_index << " to files." << std::endl;
        exit(EXIT_FAILURE);
    }
}

std::vector<std::vector<float>> MultiHeadAttention::forward(
    const std::vector<std::vector<float>> &query,
    const std::vector<std::vector<float>> &key,
    const std::vector<std::vector<float>> &value,
    const std::vector<int>& padding_mask,
    int head_number
)

{
  
    if(head_number == 0)// We only need to Cache the full sized (unsplited) input vectors once
    {
        // Cache the query, key, and value inputs for backpropagation
        query_cache = query;
        key_cache   = key;
        value_cache = value;
    }

    // Compute segment size based on number of heads
    size_t segment_size = query[0].size() / num_heads;

    // Split input vectors query, key, and value into smaller segments for this head
    std::vector<std::vector<float>> query_local_single_head(query.size(), std::vector<float>(segment_size));
    std::vector<std::vector<float>> key_local_single_head(key.size(), std::vector<float>(segment_size));
    std::vector<std::vector<float>> value_local_single_head(value.size(), std::vector<float>(segment_size));

    // Split weight matrices into smaller segments for this head
    std::vector<std::vector<float>> weights_q_local_single_head(segment_size, std::vector<float>(segment_size));
    std::vector<std::vector<float>> weights_k_local_single_head(segment_size, std::vector<float>(segment_size));
    std::vector<std::vector<float>> weights_v_local_single_head(segment_size, std::vector<float>(segment_size));

    for (size_t i = 0; i < query.size(); ++i) {
        for (size_t j = 0; j < segment_size; ++j) {
            int index = head_number * segment_size + j;
            query_local_single_head[i][j] = query[i][index];
            key_local_single_head[i][j] = key[i][index];
            value_local_single_head[i][j] = value[i][index];
        }
    }

    for (size_t i = 0; i < segment_size; ++i) {
        for (size_t j = 0; j < segment_size; ++j) {
            int index_col = head_number * segment_size + j;
            int index_row = head_number * segment_size + i;
            weights_q_local_single_head[i][j] = weights_q[index_row][index_col];
            weights_k_local_single_head[i][j] = weights_k[index_row][index_col];
            weights_v_local_single_head[i][j] = weights_v[index_row][index_col];
        }
    }

    // 1. Linear transformations for Q, K, V for this head
    auto Q = Utils::matmul(query_local_single_head, weights_q_local_single_head);
    auto K = Utils::matmul(key_local_single_head, weights_k_local_single_head);
    auto V = Utils::matmul(value_local_single_head, weights_v_local_single_head);

    // 2. Scaled dot-product attention
    std::vector<std::vector<float>> attention_output;
#ifndef PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION
    attention_output = scaled_dot_product_attention(Q, K, V, padding_mask);
#else
    attention_output = scaled_dot_product_attention_with_printout(Q, K, V);
#endif

    // Merge attention output back into original format
    std::vector<std::vector<float>> merged_attention_output(attention_output.size(), std::vector<float>(query[0].size(), 0.0f));

    for (size_t i = 0; i < attention_output.size(); ++i) {
        for (size_t j = 0; j < segment_size; ++j) {
            int index = head_number * segment_size + j;
            merged_attention_output[i][index] = attention_output[i][j];
        }
    }

    return merged_attention_output;
}

#include <stdexcept> // for std::out_of_range

float MultiHeadAttention::read_weight(const std::string& matrix_type, int row, int col) const
{
    // Decide which matrix to read from
    const std::vector<std::vector<float>>* target_matrix = nullptr;

    if (matrix_type == "Q") {
        target_matrix = &weights_q;
    } else if (matrix_type == "K") {
        target_matrix = &weights_k;
    } else if (matrix_type == "V") {
        target_matrix = &weights_v;
    } else {
        throw std::invalid_argument("Invalid matrix_type. Must be one of {\"Q\", \"K\", \"V\"}.");
    }

    // Safety check for out-of-range
    if (row < 0 || row >= static_cast<int>(target_matrix->size())) {
        throw std::out_of_range("Row index out of range in read_weight()");
    }
    if (col < 0 || col >= static_cast<int>((*target_matrix)[row].size())) {
        throw std::out_of_range("Column index out of range in read_weight()");
    }

    return (*target_matrix)[row][col];
}


void MultiHeadAttention::update_weights()
{
    // Example: read from some config or define here
    float learning_rate = GLOBAL_ATTENTION_learning_rate; 
    float momentum      = GLOBAL_ATTENTION_momentum;  

    // Update weights_q
    for (size_t i = 0; i < weights_q.size(); i++) {
        for (size_t j = 0; j < weights_q[0].size(); j++) {
            // velocity_q = momentum * velocity_q + grad
            velocity_q[i][j] = momentum * velocity_q[i][j] 
                               + learning_rate * grad_weights_q[i][j];
            // w_q -= velocity_q
            weights_q[i][j] -= velocity_q[i][j];
            // Optionally reset the grad to zero
            grad_weights_q[i][j] = 0.0f;
        }
    }

    // Update weights_k
    for (size_t i = 0; i < weights_k.size(); i++) {
        for (size_t j = 0; j < weights_k[0].size(); j++) {
            velocity_k[i][j] = momentum * velocity_k[i][j] 
                               + learning_rate * grad_weights_k[i][j];
            weights_k[i][j] -= velocity_k[i][j];
            grad_weights_k[i][j] = 0.0f;
        }
    }

    // Update weights_v
    for (size_t i = 0; i < weights_v.size(); i++) {
        for (size_t j = 0; j < weights_v[0].size(); j++) {
            velocity_v[i][j] = momentum * velocity_v[i][j]
                               + learning_rate * grad_weights_v[i][j];
            weights_v[i][j] -= velocity_v[i][j];
            grad_weights_v[i][j] = 0.0f;
        }
    }
}

std::vector<std::vector<float>> MultiHeadAttention::scaled_dot_product_attention(
    const std::vector<std::vector<float>>& query,
    const std::vector<std::vector<float>>& key,
    const std::vector<std::vector<float>>& value,
    const std::vector<int>& padding_mask)
{
    // 1. Compute QK^T
    auto scores = Utils::matmul(query, Utils::transpose(key));
    // 2. Scale scores by sqrt(d_k)
    float scale_factor = std::sqrt(static_cast<float>(key[0].size()));
    for (size_t i = 0; i < scores.size(); ++i) {
        for (size_t j = 0; j < scores[0].size(); ++j) {
            scores[i][j] /= scale_factor;
        }
    }
    // 3. Apply masking to prevent attending to future tokens
    // Apply masking to prevent attending to future tokens and padding
    for (size_t i = 0; i < scores.size(); ++i) {
        for (size_t j = 0; j < scores[i].size(); ++j) {
            if (j > i || padding_mask[j] == 0) {
                scores[i][j] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    // 4. Apply softmax to scores
    // Softmax
    for (size_t i = 0; i < scores.size(); ++i) {
        if(inference_mode == true)
        {
            scores[i] = Utils::softmax(scores[i]);
        }
        else
        {
            scores[i] = Utils::denoise_softmax(scores[i], 0.05);
        }
    }
    // 5. Multiply scores with V
    // Store final attention distribution for backprop
    attention_probs_cache = scores;
    auto output = Utils::matmul(scores, value);
    return output;
}

