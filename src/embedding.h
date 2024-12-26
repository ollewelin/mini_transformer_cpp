#include <vector>
class Embedding {
public:
    Embedding(int vocab_size, int d_model);
    std::vector<std::vector<float>> forward(const std::vector<int>& input);

private:
    std::vector<std::vector<float>> embedding_matrix;
};
