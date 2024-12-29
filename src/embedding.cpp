#include "embedding.h"
#include <cstdlib>  // For rand()
#include <ctime>    // For seeding rand()

// Constructor
Embedding::Embedding(int vocab_size, int d_model) {
    // Seed the random number generator
    std::srand(std::time(0));
    
    // Initialize embedding matrix with random values
    embedding_matrix = std::vector<std::vector<float>>(vocab_size, std::vector<float>(d_model));
    for (int i = 0; i < vocab_size; ++i) {
        for (int j = 0; j < d_model; ++j) {
            embedding_matrix[i][j] = static_cast<float>(std::rand()) / RAND_MAX; // Random float between 0 and 1
        }
    }
}

// Forward pass
std::vector<std::vector<float>> Embedding::forward(const std::vector<int>& input) {
    std::vector<std::vector<float>> result;

    // Fetch the embedding vector for each token ID
    for (int token_id : input) {
        result.push_back(embedding_matrix[token_id]);
    }

    return result;
}
