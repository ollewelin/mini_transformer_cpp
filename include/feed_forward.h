#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdexcept>
#include <cstdlib>
#include <ctime>
#include "config.h"

class FeedForward {
public:
    FeedForward(int d_model, int d_ff, bool load_parameters_yes_no, int layer_index);

    std::vector<std::vector<float>> forward(const std::vector<std::vector<float>>& input);

private:
    std::vector<std::vector<float>> weights1; // First linear layer weights
    std::vector<std::vector<float>> weights2; // Second linear layer weights
};
#endif // FEED_FORWARD_H