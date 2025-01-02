#include <vector>
#include "config.h"
class FeedForward {
public:
    FeedForward(int d_model, int d_ff);
    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);

private:
    std::vector<std::vector<float>> weights1;
    std::vector<std::vector<float>> weights2;
};
