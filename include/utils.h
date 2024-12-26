#include <vector>
namespace Utils {
    std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b);
    std::vector<std::vector<float>> transpose(const std::vector<std::vector<float>>& matrix);
    std::vector<float> softmax(const std::vector<float>& input);
};
