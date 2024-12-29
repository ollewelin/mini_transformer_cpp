#include "transformer.h"
#include <iostream>

using namespace std;

int main() {

    cout << "Transformer test in miniformat C/C++ no use of ML (Machine Learning) library" << endl;

    // Define parameters
    int vocab_size = 5000;
    int d_model = 128; // The "resolution" of the positional encoding space. 
                   // Like a meter stick with 128 evenly spaced lines, 
                   // this determines how finely token positions are encoded.
    int num_heads = 8;
    int d_ff = 256;
    int num_layers = 6;
    int max_len = 64; // Maximum sequence length (number of tokens in a single input)

    std::cout << "Do you want to load an existing model parameter with embedding matrix from a file? (Y/N, y/n): ";
    std::string choice;
    std::cin >> choice;

    bool load_parameters_yes_no = false;
    if (choice == "Y" || choice == "y" ) {
        load_parameters_yes_no = true;// Load from file
    } else {
        load_parameters_yes_no = false; // Random initialization
    }

    // Create transformer
    Transformer transformer(vocab_size, d_model, max_len, num_heads, d_ff, num_layers, load_parameters_yes_no);
    
    // Sample input (token IDs)
    std::vector<int> input = {1, 2, 3, 4, 5};

    if (input.size() > max_len) {
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

