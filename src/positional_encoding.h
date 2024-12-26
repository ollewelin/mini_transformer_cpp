#include <vector>
class PositionalEncoding {
public:
    PositionalEncoding(int max_len, int d_model);
    std::vector<std::vector<float>> add_positional_encoding(const std::vector<std::vector<float>>& input);

private:
    std::vector<std::vector<float>> pos_encoding;
};
