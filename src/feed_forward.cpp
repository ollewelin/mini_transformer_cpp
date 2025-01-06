#include "feed_forward.h"
#include "utils.h"
const std::string FeedForward::file_prefix_feed_forward_weights = "ffd_weight_layer_";
FeedForward::FeedForward(int d_model, int d_ff, bool load_parameters_yes_no, int layer_index)
    : weights1(d_model, std::vector<float>(d_ff, 0.0f)),
      weights2(d_ff, std::vector<float>(d_model, 0.0f))
{
    const std::string weights_file = file_prefix_feed_forward_weights + std::to_string(layer_index) + ".bin";
    bool loaded = false;

    if (load_parameters_yes_no) {
        std::ifstream file(weights_file, std::ios::binary);
        if (file.is_open()) {
            // Load weights1
            for (auto& row : weights1) {
                file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            // Load weights2
            for (auto& row : weights2) {
                file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
            }
            file.close();
            std::cout << "FeedForward weights for layer " << layer_index << " loaded from file.\n";
            loaded = true;
        } else {
            std::cerr << "Warning: Could not open " << weights_file << ". Falling back to random initialization.\n";
        }
    }

    if (!loaded) {
        std::srand(std::time(0));
        float scale1 = std::sqrt(2.0f / d_model);
        float scale2 = std::sqrt(2.0f / d_ff);

        // Randomly initialize weights1
        for (auto& row : weights1) {
            for (auto& val : row) {
                val = scale1 * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;
            }
        }

        // Randomly initialize weights2
        for (auto& row : weights2) {
            for (auto& val : row) {
                val = scale2 * (static_cast<float>(std::rand()) / RAND_MAX - 0.5f) * 2.0f;
            }
        }
    }
}

void FeedForward::save_weights(int layer_index) {
    const std::string weights_file = file_prefix_feed_forward_weights + std::to_string(layer_index) + ".bin";
    std::ofstream file(weights_file, std::ios::binary);
    if (file.is_open()) {
        for (const auto& row : weights1) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
        for (const auto& row : weights2) {
            file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
        }
        file.close();
        std::cout << "FeedForward weights for layer " << layer_index << " initialized and saved to file.\n";
    } else {
        std::cerr << "Error: Could not save FeedForward weights to file.\n";
        exit(EXIT_FAILURE);
    }
}
std::vector<std::vector<float>> FeedForward::forward(const std::vector<std::vector<float>>& input)
{
    // Step 1: Linear transformation with weights1
    auto hidden = Utils::matmul(input, weights1);

    // Step 2: Apply ReLU activation
    for (auto& row : hidden) {
        for (auto& val : row) {
            //val = std::max(0.0f, val); // ReLU activation
            val = Utils::leaky_relu(val); // ReLU Leaky activation
        }
    }

    // Step 3: Linear transformation with weights2
    auto output = Utils::matmul(hidden, weights2);

    return output;
}

std::vector<std::vector<float>> FeedForward::backward(const std::vector<std::vector<float>>& grad_output) {
    // Step 1: Compute gradients for weights2
    auto weights2_transposed = Utils::transpose(weights2);
    auto grad_hidden = Utils::matmul(grad_output, weights2_transposed);

    // Step 2: Apply derivative of ReLU activation to grad_hidden
    for (size_t i = 0; i < grad_hidden.size(); ++i) {
        for (size_t j = 0; j < grad_hidden[i].size(); ++j) {
            grad_hidden[i][j] *= (grad_hidden[i][j] > 0) ? 1.0f : GLOBAL_LEAKY_SLOPE;
        }
    }

    // Step 3: Compute gradients for weights1
    auto weights1_transposed = Utils::transpose(weights1);
    auto grad_input = Utils::matmul(grad_hidden, weights1_transposed);

    return grad_input;
}

