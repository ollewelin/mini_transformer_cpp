#include "transformer.h"
#include <iostream>

int main() {
    // Define parameters
    int vocab_size = 5000;
    int d_model = 128;
    int num_heads = 8;
    int d_ff = 256;
    int num_layers = 6;
    int max_len = 64; // Maximum sequence length (number of tokens in a single input)


    // Create transformer
    Transformer transformer(vocab_size, d_model, max_len, num_heads, d_ff, num_layers);

    // Sample input (token IDs)
    std::vector<int> input = {1, 2, 3, 4, 5};

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

