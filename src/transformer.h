#include <vector>
#include "embedding.h"
#include "positional_encoding.h"
#include "attention.h"
#include "feed_forward.h"

class Transformer
{
public:
    Transformer(int vocab_size, int d_model, int max_len, int num_heads, int d_ff, int num_layers)
        : embedding(vocab_size, d_model), // Initialize nested object here
          pos_encoding(max_len, d_model)           // Initialize another nested object here
    {
        // Constructor body (if needed)
    };
    std::vector<std::vector<float>> forward(const std::vector<int> &input);

private:
    Embedding embedding;
    PositionalEncoding pos_encoding;
    std::vector<MultiHeadAttention> attention_layers;
    std::vector<FeedForward> feed_forward_layers;
};


