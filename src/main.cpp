#include "transformer.h"
#include <iostream>
#include "dataset.h"
#include <vector>

using namespace std;

#include "config.h"
#ifdef TEST_UTILS
#include "attention.h"
#include "utils.h"
#endif
#ifdef PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION
#include "utils.h"
#endif

#ifdef TEST_FEEDFORWARD
#include "feed_forward.h"
#endif


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

    int num_heads = 2;//8
    int d_ff = 256;
    int num_layers = 6;
    int max_len = 64; // Maximum sequence length (number of tokens in a single input)

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
        {"what", 1}, {"time", 2}, {"is", 3}, {"it", 4}, {"now", 5},
        {"how", 6}, {"are", 7}, {"you", 8}, {"doing", 9}, {"today", 10},
        {"can", 11}, {"help", 12}, {"me", 13}, {"with", 14}, {"this", 15},
        {"where", 16}, {"the", 17}, {"nearest", 18}, {"bus", 19}, {"stop", 20},
        {"why", 21}, {"sky", 22}, {"blue", 23}, {"who", 24}, {"wrote", 25},
        {"book", 26}, {"which", 27}, {"movie", 28}, {"do", 29}, {"recommend", 30},
        {"when", 31}, {"will", 32}, {"meeting", 33}, {"start", 34}, {"going", 35},
        {"to", 36}, {"rain", 37}, {"could", 38}, {"explain", 39}, {"that", 40},
        {"again", 41}, {"three", 42}, {"oclock", 43}, {"am", 44}, {"well", 45},
        {"thank", 46}, {"yes", 47}, {"i", 48}, {"help", 49}, {"light", 50},
        {"scattering", 51}, {"jane", 52}, {"austen", 53}, {"inception", 54},
        {"ten", 55}, {"minutes", 56}, {"sure", 57}, {"later", 58}
    };

    // Prepare the dataset
    std::vector<std::vector<int>> data;
    std::vector<int> labels;
    prepare_dataset(data, labels, vocab);

    // Display tokenized sentences and their labels
    std::cout << "Tokenized Dataset:\n";
    for (size_t i = 0; i < data.size(); ++i) {
        std::cout << (labels[i] == 0 ? "Question: " : "Answer: ");
        for (int token : data[i]) {
            std::cout << token << " ";
        }
        std::cout << "\n";
    }

    //================ Set up the transformer ===============
    vocab_size = vocab.size(); // Dynamically set to the actual vocabulary size
    d_model = 6;
    // Create transformer
    Transformer transformer(vocab_size, d_model, max_len, num_heads, d_ff, num_layers, load_parameters_yes_no);

    // Input with padding
    std::vector<int> input = {1, 2, 3, 0, 0}; // Tokens with [PAD]

    // Padding mask
    std::vector<int> padding_mask = {1, 1, 1, 0, 0};

    if (input.size() > (unsigned int)max_len)
    {
        std::cerr << "Error: Input sequence length exceeds maximum allowed length of " << max_len << "." << std::endl;
        return 1;
    }

    std::cout << "input vector : ";
    for (int val : input) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Forward pass
    auto output = transformer.forward(input, padding_mask);

    // Print output
    int row_nr=0;
    int col_nr=0;
    for (const auto& row : output) {
        std::cout << "output Row: " << row_nr << endl;
        row_nr++;
        for (float val : row) {
            std::cout << " col_nr: " << col_nr << " data: "<< val <<endl;
            col_nr++;
        }
        std::cout << "\n";
        std::cout << "\n";
    }

    transformer.save_layer_norm_weights();
    transformer.save_embedding_matrix();
    transformer.save_attention_weights();
    transformer.save_feed_forward_weights();
#endif

    return 0;

}

