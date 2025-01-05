#include "transformer.h"
#include <iostream>
#include "dataset.h"
#include <vector>
#include "utils.h"
using namespace std;

#include "config.h"
#ifdef TEST_UTILS
#include "attention.h"
#endif

#ifdef TEST_FEEDFORWARD
#include "feed_forward.h"
#endif

#include <algorithm> // For Fisher-Yates shuffle
#include <random>    // For random number generation
#include <chrono>    // For seeding random number generator

// Function to shuffle dataset
void fisher_yates_shuffle(std::vector<std::vector<int>>& dataset, std::vector<int>& labels) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);

    for (size_t i = dataset.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);

        std::swap(dataset[i], dataset[j]);
        std::swap(labels[i], labels[j]);
    }
}

// Function to pad a sequence to `max_len`
std::vector<int> pad_sequence(const std::vector<int>& sequence, int max_len) {
    std::vector<int> padded_sequence = sequence;
    if (padded_sequence.size() < (size_t)max_len) {
        padded_sequence.resize(max_len, 0); // Pad with 0s (assumed [PAD] token)
    }
    return padded_sequence;
}

// Function to create padding mask
std::vector<int> create_padding_mask(const std::vector<int>& sequence, int max_len) {
    std::vector<int> mask(max_len, 0);
    for (size_t i = 0; i < sequence.size(); ++i) {
        if (sequence[i] != 0) { // Assume non-zero tokens are valid
            mask[i] = 1;
        }
    }
    return mask;
}

// Mean Pooling
std::vector<float> mean_pooling(const std::vector<std::vector<float>>& output) {
    std::vector<float> pooled(output[0].size(), 0.0f);
    for (const auto& row : output) {
        for (size_t i = 0; i < row.size(); ++i) {
            pooled[i] += row[i];
        }
    }
    for (float& val : pooled) {
        val /= output.size();
    }
    return pooled;
}
//Final Classification Layer 
std::vector<float> linear_layer(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias) {
    std::vector<float> output(weights[0].size(), 0.0f);

    for (size_t i = 0; i < weights[0].size(); ++i) { // For each output category
        for (size_t j = 0; j < input.size(); ++j) {  // For each input dimension
            output[i] += input[j] * weights[j][i];
        }
        output[i] += bias[i]; // Add bias term
    }

    return output;
}
//Final Classification Softmax Layer
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end()); // For numerical stability
    float sum_exp = 0.0f;

    for (float logit : logits) {
        sum_exp += std::exp(logit - max_logit);
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit) / sum_exp;
    }

    return probabilities;
}
#include <fstream>

// Save function for the final layer
void save_final_layer_weights(const std::vector<std::vector<float>>& weights, const std::vector<float>& bias) {
    std::ofstream file("final_layer_weight.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file final_layer_weight.bin for saving." << std::endl;
        return;
    }

    // Save weights
    for (const auto& row : weights) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }

    // Save bias
    file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(float));
    file.close();
    std::cout << "Final layer weights saved to final_layer_weight.bin." << std::endl;
}

// Load function for the final layer
bool load_final_layer_weights(std::vector<std::vector<float>>& weights, std::vector<float>& bias) {
    std::ifstream file("final_layer_weight.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file final_layer_weight.bin for loading. Falling back to random initialization." << std::endl;
        return false;
    }

    // Load weights
    for (auto& row : weights) {
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    }

    // Load bias
    file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(float));
    file.close();
    std::cout << "Final layer weights loaded from final_layer_weight.bin." << std::endl;
    return true;
}


int main() {
    cout << "========================================================================================================" << endl;
    cout << "Transformer Test in Mini Format (C/C++) - No Use of ML Libraries" << endl;
    cout << "The goal is to build and understand the Transformer algorithm from scratch using pure C++." << endl;
    cout << "========================================================================================================" << endl;
    cout << endl;
    std::cout << "Do you want to load an existing model parameter with embedding matrix from a file? (Y/N, y/n): ";
    std::string choice;
    std::cin >> choice;

    bool load_parameters_yes_no = false;
    if (choice == "/Y" || choice == "y")
    {
        load_parameters_yes_no = true; // Load from file
    }
    else
    {
        load_parameters_yes_no = false; // Random initialization
    }

#ifdef TEST_FEEDFORWARD
    cout << "==================== Test: FeedForward ====================\n";

    // Define dimensions for the test
    int d_model_ffd_test = 4;  // Input dimensionality
    int d_ff_ffd_test = 6;     // Hidden layer dimensionality
    int layer_index_ffd_test = 0;

    // Create a FeedForward object
    FeedForward feed_forward(d_model_ffd_test, d_ff_ffd_test, load_parameters_yes_no, layer_index_ffd_test);

    // Define a small input matrix (e.g., 2 tokens with d_model dimensions)
    std::vector<std::vector<float>> input_ffd_test = {
        {1.0, 2.0, 3.0, 4.0},
        {0.5, 0.6, 0.7, 0.8}
    };

    cout << "input_ffd_test matrix (shape: " << input_ffd_test.size() << " x " << input_ffd_test[0].size() << "):\n";
    Utils::print_matrix(input_ffd_test);

    // Forward pass through the FeedForward network
    auto output_ffd_test = feed_forward.forward(input_ffd_test);

    // Print the output_ffd_test
    cout << "Output matrix (shape: " << output_ffd_test.size() << " x " << output_ffd_test[0].size() << "):\n";
    Utils::print_matrix(output_ffd_test);

    cout << "==========================================================\n";
#endif

#ifdef PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION
    // Make some PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION with small input matrix tests 
    // to understand one single layer of attention head in operation.
    cout << "==================== Test: Single Attention Layer ====================\n";

    // Input matrices
    std::vector<std::vector<float>> Q = {
        {1.0, 0.5, 0.1, 0.01}, // Query vector for token 1
        {0.2, 1.3, 0.2, 0.02}, // Query vector for token 2
        {1.2, 2.3, 3.2, 4.11}  // Query vector for token 3
    };
    std::vector<std::vector<float>> K = {
        {0.8, 0.3, 0.3, 0.03}, // Key vector for token 1
        {0.1, 0.9, 0.4, 0.04}, // Key vector for token 2
        {0.2, 0.3, 3.0, 1.11}  // Key vector for token 3
    };
    std::vector<std::vector<float>> V = {
        {1.2, 0.7, 0.5, 0.05}, // Value vector for token 1
        {0.5, 0.4, 0.6, 0.06}, // Value vector for token 2
        {2.2, 1.3, 0.0, 3.11}  // Value vector for token 3
    };

    // Check matrix sizes using Utils
    Utils::check_matrices(Q, K, V);

    // Simplified test setup
    
    int d_model_test = Q[0].size();               // Total embedding dimension
    int num_heads_test = 1;                  // Number of attention heads
    int d_k = d_model_test / num_heads_test; // Dimensionality of Q and K
    int d_v = d_model_test / num_heads_test; // Dimensionality of V

    cout << "The resolution of the positional encoding and embedding space, d_model: " << d_model_test << endl;
    MultiHeadAttention attention_layer_printout(d_model_test, 1, false, 0); // d_model=4, num_heads=1, no load, layer_index=0

    cout << "\n=== Relationship Between d_model, num_heads, and Matrix Dimensions ===\n";
    cout << "d_model (total embedding dimension): " << d_model_test << "\n";
    cout << "num_heads (number of attention heads): " << num_heads_test << "\n";
    cout << "d_k (key/query dimension per head): " << d_k << "\n";
    cout << "d_v (value dimension per head): " << d_v << "\n";

    cout << "\nExplanation:\n";
    cout << "- The total embedding dimension (d_model) is divided among all attention heads.\n";
    cout << "- With num_heads = 1, each head gets the full d_model, so d_k = d_model / num_heads = " << d_k << ".\n";
    cout << "- Similarly, d_v = d_model / num_heads = " << d_v << ".\n";
    cout << "In this case, each token is represented with " << d_k << " dimensions in Q and K, and "
         << d_v << " dimensions in V.\n";

    cout << "\n=== Hard coded Test Input Matrices ===\n";

    // Print matrices
    cout << "\nInput Q (Query):\n";
    Utils::print_matrix(Q);
    cout << "Each row represents a token, and each column represents one of the " << d_k << " dimensions of the query vector.\n";

    cout << "\nInput K (Key):\n";
    Utils::print_matrix(K);
    cout << "Each row represents a token, and each column represents one of the " << d_k << " dimensions of the key vector.\n";

    cout << "\nInput V (Value):\n";
    Utils::print_matrix(V);
    cout << "Each row represents a token, and each column represents one of the " << d_v << " dimensions of the value vector.\n";

    cout << "\nSummary:\n";
    cout << "- Q and K have " << d_k << " columns because they encode positional and content-related similarities.\n";
    cout << "- V has " << d_v << " columns because it contains the actual token content to be weighted and combined.\n";
    cout << "=====================================================================\n";
    
    // Call scaled_dot_product_attention_with_printout for testing
    auto attention_output_printout = attention_layer_printout.scaled_dot_product_attention_with_printout(Q, K, V);

    cout << "=====================================================================\n";

#else

    // Define parameters
    int vocab_size = 5000;
    int d_model = 128; // The "resolution" of the positional encoding and embedding space. 
                    // Think of it like a meter stick with 128 evenly spaced lines: 
                    // this determines how finely the meaning of a token can be represented
                    // across multiple dimensions.
                    //
                    // Each token (word or sub-word) is not just an isolated entity but carries 
                    // a representation that heavily depends on its position and relationships 
                    // to other tokens in the context. For example, the word "bank" could 
                    // mean "riverbank" or "financial bank," and its meaning is influenced 
                    // by neighboring words.
                    //
                    // In this context, "d_model" defines the number of dimensions (features) 
                    // used to represent these relationships. Higher d_model provides a finer 
                    // "resolution," allowing the model to encode more complex interactions 
                    // and associations across the sequence. 
                    //
                    // Increasing d_model expands the range of nuances and relationships that 
                    // the model can capture, enabling it to differentiate subtle differences 
                    // in meaning based on positional and contextual variations in the input 
                    // sequence.
                    //
                    // However, higher d_model also increases computational complexity and 
                    // the risk of overfitting for small datasets, so a balance is needed.

    int num_heads = 2;// 8
    int d_ff = 256;   // d_ff: Dimensionality of the hidden layer in the feed-forward network.
                      //       Each feed-forward network in the transformer consists of two linear layers:
                      //       - The first layer expands the input dimensionality (d_model) to a larger hidden size (d_ff).
                      //       - The second layer projects the hidden layer back down to the original dimensionality (d_model).
                      //       This expansion allows the model to learn richer, non-linear representations
                      //       by operating in a higher-dimensional space during the intermediate steps.
                      //
                      //       Typical values for d_ff are 2-4 times larger than d_model.
                      //       For example:
                      //         d_model = 128, d_ff = 256 or d_ff = 512.
                      //       This ratio balances the model's capacity with computational efficiency.
    int num_layers = 6;
    int max_len = 10; //64  Maximum sequence length (number of tokens in a single input)

#ifdef TEST_UTILS

    cout << "Test utils functions here: " << endl;

    // test utils funcftions
    // Test 1: Matrix Multiplication
    vector<vector<float>> mat1 = {{10.0, 20.0, 30.0}, {40.0, 50.0, 60.0}};
    vector<vector<float>> mat2 = {{1.0, 2.0}, {3.0, 4.0}, {5.0, 6.0}};
    vector<vector<float>> matmul_result = Utils::matmul(mat1, mat2);

    cout << "\nMatrix Multiplication Result:" << endl;
    for (const auto &row : matmul_result)
    {
        for (float val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }

    // Test 2: Matrix Transpose
    vector<vector<float>> transpose_result = Utils::transpose(mat1);

    cout << "\nMatrix Transpose Result:" << endl;
    for (const auto &row : transpose_result)
    {
        for (float val : row)
        {
            cout << val << " ";
        }
        cout << endl;
    }

    // Test 3: Softmax
    vector<float> logits = {2.0, 1.0, 0.1};
    vector<float> softmax_result = Utils::softmax(logits);

    cout << "\nSoftmax Result:" << endl;
    for (float val : softmax_result)
    {
        cout << val << " ";
    }
    cout << endl;

//
#endif
#ifdef TEST_ATTENTION

   cout << "Testing MultiHeadAttention functions..." << endl;

    // Test Inputs
    vector<vector<float>> query = {{1.0, 0.0}, {0.0, 1.0}};
    vector<vector<float>> key = {{1.0, 2.0}, {0.0, 3.0}};
    vector<vector<float>> value = {{4.0, 5.0}, {6.0, 7.0}};
    // Padding mask
    std::vector<int> padding_mask_test = {1, 1, 1};

    // Initialize MultiHeadAttention with 2 dimensions and 1 head (simplest case)
    MultiHeadAttention attention(2, 1, load_parameters_yes_no, num_layers);

    // Manually set weights for testing (simplified identity weights)
    attention.weights_q = {{1.0, 0.0}, {0.0, 1.0}};
    attention.weights_k = {{1.0, 0.0}, {0.0, 1.0}};
    attention.weights_v = {{1.0, 0.0}, {0.0, 1.0}};

    // Test Scaled Dot-Product Attention
    cout << "\nTesting Scaled Dot-Product Attention:" << endl;
    auto attention_output = attention.scaled_dot_product_attention(query, key, value, padding_mask_test);

    cout << "Attention Output:" << endl;
    for (const auto& row : attention_output) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    // Test Full Forward Pass
    cout << "\nTesting Full Forward Pass:" << endl;
    auto forward_output = attention.forward(query, key, value, padding_mask_test);

    cout << "Forward Output:" << endl;
    for (const auto& row : forward_output) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
#endif

    // Define a simple vocabulary
    std::unordered_map<std::string, int> vocab = {
        {"what", 0}, {"time", 1}, {"is", 2}, {"it", 3}, {"now", 4},
        {"how", 5}, {"are", 6}, {"you", 7}, {"doing", 8}, {"today", 9},
        {"can", 10}, {"help", 11}, {"me", 12}, {"with", 13}, {"this", 14},
        {"where", 15}, {"the", 16}, {"nearest", 17}, {"bus", 18}, {"stop", 19},
        {"why", 20}, {"sky", 21}, {"blue", 22}, {"who", 23}, {"wrote", 24},
        {"book", 25}, {"which", 26}, {"movie", 27}, {"do", 28}, {"recommend", 29},
        {"when", 30}, {"will", 31}, {"meeting", 32}, {"start", 33}, {"going", 34},
        {"to", 35}, {"rain", 36}, {"could", 37}, {"explain", 38}, {"that", 39},
        {"again", 40}, {"three", 41}, {"oclock", 42}, {"am", 43}, {"well", 44},
        {"thank", 45}, {"yes", 46}, {"i", 47}, {"light", 48}, {"scattering", 49},
        {"jane", 50}, {"austen", 51}, {"inception", 52}, {"ten", 53},
        {"minutes", 54}, {"sure", 55}, {"later", 56}
    };
    if (!Utils::check_vocabs(vocab)) {
        std::cerr << "Vocabulary validation failed.\n";
        return 1; // Exit with error
    }
    std::cout << "Vocabulary validation succeeded.\n";
    // Prepare the dataset
    std::vector<std::vector<int>> dataset_2D;
    std::vector<int> labels;
    prepare_dataset(dataset_2D, labels, vocab);

    // Display tokenized sentences and their labels
    std::cout << "Tokenized Dataset:\n";
    for (size_t i = 0; i < dataset_2D.size(); ++i) {
        std::cout << (labels[i] == 0 ? "Question: " : "Answer: ");
        for (int token : dataset_2D[i]) {
            std::cout << token << " ";
        }
        std::cout << "\n";
    }

    // ================== Set up the transformer ==================
    vocab_size = vocab.size(); // Dynamically set to the actual vocabulary size
    cout << "vocab_size = " << vocab_size << endl;

    d_model = 6;
    d_ff = 24;

    // Initialize final layer weights and bias
    int num_categories = 2; // Number of output categories (Question/Answer)

    std::vector<std::vector<float>> final_weights(d_model, std::vector<float>(num_categories, 0.0f));
    std::vector<float> final_bias(num_categories, 0.0f);

    if (load_parameters_yes_no) {
        if (!load_final_layer_weights(final_weights, final_bias)) {
            // Fall back to random initialization if loading fails
            std::srand(std::time(0));
            for (auto& row : final_weights) {
                for (auto& val : row) {
                    val = static_cast<float>(std::rand()) / RAND_MAX; // Random values between 0 and 1
                }
            }
            for (auto& val : final_bias) {
                val = static_cast<float>(std::rand()) / RAND_MAX;
            }
        }
    } else {
        // Random initialization
        std::srand(std::time(0));
        for (auto& row : final_weights) {
            for (auto& val : row) {
                val = static_cast<float>(std::rand()) / RAND_MAX;
            }
        }
        for (auto& val : final_bias) {
            val = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    // Create transformer
    Transformer transformer(vocab_size, d_model, max_len, num_heads, d_ff, num_layers, load_parameters_yes_no);

    // ============== Training loop ===================
    int epochs = 10;

    for (int epoch = 1; epoch <= epochs; ++epoch) {
        std::cout << "Epoch " << epoch << " / " << epochs << "\n";

        // Shuffle dataset
        fisher_yates_shuffle(dataset_2D, labels);

        for (size_t i = 0; i < dataset_2D.size(); ++i) {
            // Prepare input and padding mask
            auto padded_input = pad_sequence(dataset_2D[i], max_len);
            auto padding_mask = create_padding_mask(dataset_2D[i], max_len);

            // Forward pass through transformer
            auto output = transformer.forward(padding_mask, padding_mask);

            // Reduce transformer output (e.g., by mean pooling)
            std::vector<float> pooled_output = mean_pooling(output);

            // Apply final classification layer
            std::vector<float> logits = linear_layer(pooled_output, final_weights, final_bias);
            std::vector<float> probabilities = softmax(logits);

            // Print input and probabilities for debugging
            std::cout << "Input: ";
            for (int token : padded_input) {
                std::cout << token << " ";
            }
            std::cout << "\nProbabilities: ";
            for (float prob : probabilities) {
                std::cout << prob << " ";
            }
            
            std::cout << "\n";
        }
    }
    //========================== End training loop ===================

    // Save final layer weights (optional)
    save_final_layer_weights(final_weights, final_bias);
    transformer.save_layer_norm_weights();
    transformer.save_embedding_matrix();
    transformer.save_attention_weights();
    transformer.save_feed_forward_weights();    


#endif

    return 0;

}

