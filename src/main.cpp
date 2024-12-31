#include "transformer.h"
#include <iostream>

#include <vector>

using namespace std;

//#define TEST_xxxxx will Test and validate smaller parts of the code
//#define TEST_UTILS
//#define TEST_ATTENTION
//#include "attention.h"
//#include "utils.h"

int main() {
cout << "========================================================================================================" << endl;
cout << "Transformer Test in Mini Format (C/C++) - No Use of ML Libraries" << endl;
cout << "This mini Transformer project is heavily guided and inspired by AI-powered assistance (Transformer LLM)." << endl;
cout << "The goal is to build and understand the Transformer algorithm from scratch using pure C++." << endl;
cout << "========================================================================================================" << endl;
cout << endl;

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

    int num_heads = 8;
    int d_ff = 256;
    int num_layers = 6;
    int max_len = 64; // Maximum sequence length (number of tokens in a single input)

    std::cout << "Do you want to load an existing model parameter with embedding matrix from a file? (Y/N, y/n): ";
    std::string choice;
    std::cin >> choice;

    bool load_parameters_yes_no = false;
    if (choice == "/Y" || choice == "y" ) {
        load_parameters_yes_no = true;// Load from file
    } else {
        load_parameters_yes_no = false; // Random initialization
    }

#ifdef TEST_UTILS
    cout << "Test utils functions here: " << endl;

    // test utils funcftions
    // Test 1: Matrix Multiplication
    vector<vector<float>> mat1 = {{1.0, 2.0}, {3.0, 4.0}};
    vector<vector<float>> mat2 = {{5.0, 6.0}, {7.0, 8.0}};
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

    // Initialize MultiHeadAttention with 2 dimensions and 1 head (simplest case)
    MultiHeadAttention attention(2, 1, load_parameters_yes_no, num_layers);

    // Manually set weights for testing (simplified identity weights)
    attention.weights_q = {{1.0, 0.0}, {0.0, 1.0}};
    attention.weights_k = {{1.0, 0.0}, {0.0, 1.0}};
    attention.weights_v = {{1.0, 0.0}, {0.0, 1.0}};

    // Test Scaled Dot-Product Attention
    cout << "\nTesting Scaled Dot-Product Attention:" << endl;
    auto attention_output = attention.scaled_dot_product_attention(query, key, value);

    cout << "Attention Output:" << endl;
    for (const auto& row : attention_output) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << endl;
    }

    // Test Full Forward Pass
    cout << "\nTesting Full Forward Pass:" << endl;
    auto forward_output = attention.forward(query, key, value);

    cout << "Forward Output:" << endl;
    for (const auto& row : forward_output) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << endl;
    }
#endif

    // Create transformer
    Transformer transformer(vocab_size, d_model, max_len, num_heads, d_ff, num_layers, load_parameters_yes_no);
    
    // Sample input (token IDs)
    std::vector<int> input = {1, 2, 3, 4, 5};

    if (input.size() > (unsigned int)max_len) {
        std::cerr << "Error: Input sequence length exceeds maximum allowed length of " << max_len << "." << std::endl;
        return 1;
    }

    // Forward pass
    auto output = transformer.forward(input);
    
    // Print output
    for (const auto& row : output) {
        for (float val : row) {
            std::cout << val << " ";
        }
        std::cout << "\n";
    }
    return 0;
}

