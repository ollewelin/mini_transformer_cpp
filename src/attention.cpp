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
    : weights_q(d_model, std::vector<float>(d_model, 0.0f)),
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

      // Caches
      query_cache(),
      key_cache(),
      value_cache()
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
#ifdef PRINT_OUT_INIT_VECTORS
    // Print a few rows of weights_q, weights_k, and weights_v
    std::cout << "\nSample rows of weights_q for layer " << layer_index << ":" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, weights_q.size()); ++i) {
        for (float val : weights_q[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nSample rows of weights_k for layer " << layer_index << ":" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, weights_k.size()); ++i) {
        for (float val : weights_k[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\nSample rows of weights_v for layer " << layer_index << ":" << std::endl;
    for (size_t i = 0; i < std::min<size_t>(3, weights_v.size()); ++i) {
        for (float val : weights_v[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
#endif
}


std::vector<std::vector<float>> MultiHeadAttention::backward(
    const std::vector<std::vector<float>>& grad_output
) {
    //=============================================================
    // 0) Setup / Initialize local grad buffers
    //=============================================================
    // grad_output has the same shape as the 'attention_output' from forward
    // We'll need partial derivatives for Q, K, V, etc.
    std::vector<std::vector<float>> dV;      // same shape as V
    std::vector<std::vector<float>> dQ;      // same shape as Q
    std::vector<std::vector<float>> dK;      // same shape as K

    // You may want to initialize them to zeros. E.g.:
    // dV = Utils::zeros_like(value_cache * weights_v); etc.

    // Also, if you stored the final attention probs from forward:
    // "attention_probs_cache" shape = (batch_size x seq_len, seq_len)
    // We'll call it A for short in the comments.
    auto A = attention_probs_cache; // just to reference it easily

    //=============================================================
    // 1) Backprop through final matmul: output = A * V
    //    => dA = grad_output * V^T
    //    => dV = A^T * grad_output
    //=============================================================
    // dV = A^T matmul grad_output
    dV = Utils::matmul(Utils::transpose(A), grad_output);

    // partial derivative w.r.t. attention_probs A
    // dA = grad_output matmul V^T
    auto dA = Utils::matmul(grad_output, Utils::transpose(value_cache /* i.e. V */));
    // Actually, you used V = (value_cache x weights_v).
    // So you likely need 'Utils::transpose(V)' from your forward step. 
    // If you didn't store V explicitly, you can compute it again:
    auto V = Utils::matmul(value_cache, weights_v);
    // then dA = grad_output matmul transpose(V)
    dA = Utils::matmul(grad_output, Utils::transpose(V));

    //=============================================================
    // 2) Backprop through softmax: A = softmax(scores)
    //    => dScores = dA * derivative_of_softmax(scores)
    //=============================================================
    // You typically do an element-wise formula:
    // dScores[i] = A[i] * (dA[i] - sum_j(A[j]*dA[j]))
    // Implementation depends on your Utils::softmax derivative.

    std::vector<std::vector<float>> dScores = Utils::softmax_backward(dA, A);

    // (Pseudo-code: you'll implement the derivative in your Utils or directly here.)

    //=============================================================
    // 3) Backprop through 'scores = (QK^T)/sqrt(dk)' + mask
    //    => dQ = dScores matmul K  * (1/sqrt(dk))
    //    => dK = dScores^T matmul Q * (1/sqrt(dk))
    //=============================================================
    float scale_factor = 1.0f / std::sqrt(static_cast<float>(key_cache[0].size()));

    // Recompute Q = query_cache x weights_q
    auto Q = Utils::matmul(query_cache, weights_q);
    // Recompute K = key_cache   x weights_k
    auto K_ = Utils::matmul(key_cache,   weights_k);

    // dQ
    // dimension check: dScores shape == QK^T shape. Typically (batch_size x seq_len1, seq_len2)
    // So dQ = dScores matmul K_ * scale_factor
    dQ = Utils::matmul(dScores, K_);
    Utils::scale_inplace(dQ, scale_factor);

    // dK
    // dK = dScores^T matmul Q * scale_factor
    auto dK_tmp = Utils::matmul(Utils::transpose(dScores), Q);
    Utils::scale_inplace(dK_tmp, scale_factor);

    //=============================================================
    // 4) Backprop to the linear transformations:
    //    Q = query_cache x weights_q
    //    => dWeights_q = (query_cache^T) matmul dQ
    //    => dQuery     = dQ matmul (weights_q^T)
    // Similarly for K, V
    //=============================================================
    // (a) grad_weights_v
    // V = value_cache x weights_v
    // dWeights_v = (value_cache^T) x dV
    grad_weights_v = Utils::matmul(Utils::transpose(value_cache), dV);

    // If you need grad w.r.t. the original value input:
    // grad_value = dV matmul transpose(weights_v)
    // We'll return grad_value or store it if you want to backprop further.
    auto grad_value = Utils::matmul(dV, Utils::transpose(weights_v));

    // (b) grad_weights_q
    grad_weights_q = Utils::matmul(Utils::transpose(query_cache), dQ);

    // (c) grad_weights_k
    grad_weights_k = Utils::matmul(Utils::transpose(key_cache), dK_tmp);

    // (d) If you need grad w.r.t. the original query/key:
    // grad_query  = dQ      matmul transpose(weights_q)
    // grad_key    = dK_tmp  matmul transpose(weights_k)
    auto grad_query = Utils::matmul(dQ,     Utils::transpose(weights_q));
    auto grad_key   = Utils::matmul(dK_tmp, Utils::transpose(weights_k));
    
    //=============================================================
    // 5) Combine or return whichever gradient is relevant
    //=============================================================
    // Your signature returns std::vector<std::vector<float>>. 
    // Often we return grad_w.r.t "query" (or all three). 
    // But let's say you return grad_query for demonstration:
    return grad_query;
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
    const std::vector<int>& padding_mask)
{
    // Cache the query, key, and value inputs for backpropagation
    query_cache = query;
    key_cache   = key;
    value_cache = value;

    // 1. Linear transformations for Q, K, V
    auto Q = Utils::matmul(query, weights_q);
    auto K = Utils::matmul(key, weights_k);
    auto V = Utils::matmul(value, weights_v);

    // 2. Scaled dot-product attention
#ifndef PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION
    auto attention_output = scaled_dot_product_attention(Q, K, V, padding_mask);
#else
    auto attention_output = scaled_dot_product_attention_with_printout(Q, K, V);
#endif

    return attention_output;
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

#ifndef PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION  
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
        scores[i] = Utils::softmax(scores[i]);
    }
    // 5. Multiply scores with V
    // Store final attention distribution for backprop
    attention_probs_cache = scores;
    auto output = Utils::matmul(scores, value);
    return output;
}

#else // Use of PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION in config.h
std::vector<std::vector<float>> MultiHeadAttention::scaled_dot_product_attention_with_printout(
    const std::vector<std::vector<float>> &query,
    const std::vector<std::vector<float>> &key,
    const std::vector<std::vector<float>> &value)
{
    using namespace std;

    cout << "\n=== Scaled Dot-Product Attention Debug Output ===\n";

    // 1. Compute QK^T
    cout << "\nStep 1: Compute QK^T\n";
    cout << "Query (Q) matrix (shape: " << query.size() << " x " << query[0].size() << "):\n";
    Utils::print_matrix(query); // Assuming Utils has a method to print matrices
    cout << "Key (K) matrix (shape: " << key.size() << " x " << key[0].size() << "):\n";
    Utils::print_matrix(key);

    auto scores = Utils::matmul(query, Utils::transpose(key));
    cout << "QK^T (scores matrix, shape: " << scores.size() << " x " << scores[0].size() << "):\n";
    Utils::print_matrix(scores);
    cout << "Each element in this matrix represents the dot product similarity between a query vector (row) and a key vector (column).\n";
    cout << "For example:\n";
    cout << "  - scores[0][0] = dot product of Q[0] and K[0] (similarity between token 1's query and token 1's key).\n";
    cout << "  - scores[0][1] = dot product of Q[0] and K[1] (similarity between token 1's query and token 2's key).\n";
    cout << "  - scores[1][2] = dot product of Q[1] and K[2] (similarity between token 2's query and token 3's key).\n";
    cout << "Each row represents the similarity of a specific token's query with all tokens' keys, "
         << "and each column represents the similarity of all queries with a specific token's key.\n";

    // 2. Scale scores by sqrt(d_k)
    cout << "\nStep 2: Scale scores by sqrt(d_k)\n";
    float scale_factor = std::sqrt(static_cast<float>(key[0].size()));
    cout << "Scaling factor (sqrt(d_k)): " << scale_factor << endl;
    for (size_t i = 0; i < scores.size(); ++i)
    {
        for (size_t j = 0; j < scores[0].size(); ++j)
        {
            scores[i][j] /= scale_factor;
        }
    }
    cout << "Scaled scores matrix:\n";
    Utils::print_matrix(scores);
    cout << "Each score is scaled to adjust for the dimensionality of the key vectors.\n";
    
    // 3. Apply masking to prevent attending to future tokens
    cout << "\nStep 3: Apply masking to prevent attending to future tokens\n";
    for (size_t i = 0; i < scores.size(); ++i)
    {
        for (size_t j = 0; j < scores[i].size(); ++j)
        {
            if (j > i) // Mask future positions
            {
                scores[i][j] = -std::numeric_limits<float>::infinity();
            }
        }
    }
    cout << "Masked scores matrix:\n";
    Utils::print_matrix(scores);
    cout << "This matrix shows the scores after applying a mask to ensure that a token only attends to itself and earlier tokens.\n";

    // 4. Apply softmax to scores
    cout << "\nStep 4: Apply softmax to scores\n";
    for (size_t i = 0; i < scores.size(); ++i)
    {
        scores[i] = Utils::softmax(scores[i]);
    }
    cout << "Softmax applied (attention weights):\n";
    Utils::print_matrix(scores);
    cout << "Each row represents the attention distribution for a token. "
         << "The values sum to 1, showing how much each token attends to other tokens.\n";

    // 5. Multiply scores with V
    cout << "\nStep 5: Multiply scores with Value (V) matrix\n";
    cout << "Value (V) matrix (shape: " << value.size() << " x " << value[0].size() << "):\n";
    Utils::print_matrix(value);

    auto output = Utils::matmul(scores, value);
    cout << "Output matrix (shape: " << output.size() << " x " << output[0].size() << "):\n";
    Utils::print_matrix(output);
    cout << "Each row in the output matrix corresponds to the weighted sum of value vectors "
         << "for each token, based on its attention distribution.\n";


    cout << "=== End of Debug Output ===\n";
    return output;
}
#endif
