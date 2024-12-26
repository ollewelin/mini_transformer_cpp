#include "positional_encoding.h"
#include <cmath>

// Constructor
PositionalEncoding::PositionalEncoding(int max_len, int d_model) {
    pos_encoding = std::vector<std::vector<float>>(max_len, std::vector<float>(d_model, 0.0));

    for (int pos = 0; pos < max_len; ++pos) {
        for (int i = 0; i < d_model; ++i) {
            if (i % 2 == 0) {
                pos_encoding[pos][i] = std::sin(pos / std::pow(10000.0, i / (float)d_model));
            } else {
                pos_encoding[pos][i] = std::cos(pos / std::pow(10000.0, (i - 1) / (float)d_model));
            }
        }
    }
}

// Add positional encoding to input embeddings
std::vector<std::vector<float>> PositionalEncoding::add_positional_encoding(const std::vector<std::vector<float>>& input) {
    std::vector<std::vector<float>> result = input;

    for (size_t pos = 0; pos < input.size(); ++pos) {
        for (size_t i = 0; i < input[0].size(); ++i) {
            result[pos][i] += pos_encoding[pos][i];
        }
    }

    return result;
}
