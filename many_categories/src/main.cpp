#include "transformer.h"
#include <iostream>
#include "dataset.h"
#include <vector>
#include "utils.h"
#include <unordered_map>
#include <fstream>
#include <filesystem>
using namespace std;
#include "config.h"
#include <algorithm> // For Fisher-Yates shuffle
#include <random>    // For random number generation
#include <chrono>    // For seeding random number generator
#include <algorithm> // std::min

// Cross-entropy loss gradient
std::vector<float> cross_entropy_loss_gradient(const std::vector<float>& probabilities, int label) {
    std::vector<float> gradient(probabilities.size(), 0.0f);
    for (size_t i = 0; i < probabilities.size(); ++i) {
        gradient[i] = probabilities[i] - (i == static_cast<size_t>(label) ? 1.0f : 0.0f);
    }
    return gradient;
}
// Function to compute cross-entropy loss
float cross_entropy_loss(const std::vector<float>& probabilities, int label) {
    return -std::log(probabilities[label] + 1e-9); // Add small epsilon for numerical stability
}



// Function to shuffle dataset
void fisher_yates_shuffle(std::vector<std::vector<int>>& dataset, std::vector<int>& labels) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::mt19937 rng(seed);

    for (size_t i = dataset.size() - 1; i > 0; --i) {
        std::uniform_int_distribution<size_t> dist(0, i);
        size_t j = dist(rng);

        std::swap(dataset[i], dataset[j]);
        std::swap(labels[i], labels[j]);
    }
}

// Function to pad a sequence to `max_len`
std::vector<int> pad_sequence(const std::vector<int>& sequence, int max_len) {
    std::vector<int> padded_sequence = sequence;
    if (padded_sequence.size() < (size_t)max_len) {
        padded_sequence.resize(max_len, 0); // Pad with 0s (assumed [PAD] token)
    }
    return padded_sequence;
}

std::vector<int> truncate_tokens_max_len(const std::vector<int>& sequence, int max_len) 
{
    // 1) Truncate if necessary
    // Use std::min to avoid out-of-range if sequence is shorter
    std::vector<int> truncated(sequence.begin(), 
                               sequence.begin() + std::min<size_t>(sequence.size(), max_len));

    // 2) If truncated.size() < max_len, pad with zeros
    if (truncated.size() < static_cast<size_t>(max_len)) {
        truncated.resize(max_len, 0); // 0 = [PAD]
    }

    return truncated;
}
// Function to create padding mask
std::vector<int> create_padding_mask(const std::vector<int>& sequence, int max_len) {
    std::vector<int> mask(max_len, 0);
    for (size_t i = 0; i < sequence.size(); ++i) {
        if (sequence[i] != 0) { // Assume non-zero tokens are valid
            mask[i] = 1;
        }
    }
    return mask;
}

// Mean Pooling
std::vector<float> mean_pooling(const std::vector<std::vector<float>>& output) {
    std::vector<float> pooled(output[0].size(), 0.0f);
    for (const auto& row : output) {
        for (size_t i = 0; i < row.size(); ++i) {
            pooled[i] += row[i];
        }
    }
    for (float& val : pooled) {
        val /= output.size();
    }
    return pooled;
}
//output_trans, pooled_output_gradient
std::vector<std::vector<float>> mean_pooling_backward(
    const std::vector<std::vector<float>>& output_from_transformer, 
    const std::vector<float>& grad_pooled
) {
    size_t rows = output_from_transformer.size();
    size_t cols = output_from_transformer[0].size();
    std::vector<std::vector<float>> grad_output_to_transformer(rows, std::vector<float>(cols, 0.0f));

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            grad_output_to_transformer[i][j] = grad_pooled[j] / static_cast<float>(rows);
        }
    }
    return grad_output_to_transformer;
}

//Final Classification Layer 
std::vector<float> linear_layer(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias) {
    std::vector<float> output(weights[0].size(), 0.0f);

    for (size_t i = 0; i < weights[0].size(); ++i) { // For each output category
        for (size_t j = 0; j < input.size(); ++j) {  // For each input dimension
            output[i] += input[j] * weights[j][i];
        }
        output[i] += bias[i]; // Add bias term
    }

    return output;
}
//Final Classification Softmax Layer
std::vector<float> softmax(const std::vector<float>& logits) {
    std::vector<float> probabilities(logits.size());
    float max_logit = *std::max_element(logits.begin(), logits.end()); // For numerical stability
    float sum_exp = 0.0f;

    for (float logit : logits) {
        sum_exp += std::exp(logit - max_logit);
    }

    for (size_t i = 0; i < logits.size(); ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit) / sum_exp;
    }

    return probabilities;
}
#include <fstream>

// Save function for the final layer
void save_final_layer_weights(const std::vector<std::vector<float>>& weights, const std::vector<float>& bias) {
    std::ofstream file("final_layer_weight.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file final_layer_weight.bin for saving." << std::endl;
        return;
    }

    // Save weights
    for (const auto& row : weights) {
        file.write(reinterpret_cast<const char*>(row.data()), row.size() * sizeof(float));
    }

    // Save bias
    file.write(reinterpret_cast<const char*>(bias.data()), bias.size() * sizeof(float));
    file.close();
    std::cout << "Final layer weights saved to final_layer_weight.bin." << std::endl;
}

// Load function for the final layer
bool load_final_layer_weights(std::vector<std::vector<float>>& weights, std::vector<float>& bias) {
    std::ifstream file("final_layer_weight.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Warning: Could not open file final_layer_weight.bin for loading. Falling back to random initialization." << std::endl;
        return false;
    }

    // Load weights
    for (auto& row : weights) {
        file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    }

    // Load bias
    file.read(reinterpret_cast<char*>(bias.data()), bias.size() * sizeof(float));
    file.close();
    std::cout << "Final layer weights loaded from final_layer_weight.bin." << std::endl;
    return true;
}
void print_float_vector_1D(std::vector<float> float_vector_1D)
{
    for (float val : float_vector_1D)
    {
        std::cout << val << " ";
    }
    std::cout << std::endl;
}
void print_float_vector_2D(std::vector<std::vector<float>> float_vector_2D)
{
    for (const auto &row : float_vector_2D)
    {
        for (float val : row)
        {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

void print_out_probabilities(std::vector<float> probabilities, std::vector<int> padded_input)
{
    // Print probabilities for debugging
    std::cout << "Input: ";
    for (int token : padded_input)
    {
        std::cout << token << " ";
    }
    std::cout << "\nProbabilities: ";
    print_float_vector_1D(probabilities);
}

void run_prompt_mode(Transformer& transformer, int max_len, const unordered_map<string, int>& vocab, const std::vector<std::vector<float>>& weights, const std::vector<float>& bias, const std::vector<std::string>& categories) {
    string input;
    while (true) {
        cout << "\nEnter a string (or type 'exit' to quit): ";
        getline(cin, input);

        if (input == "exit") {
            cout << "Exiting mini prompt mode.\n";
            break;
        }

        vector<int> tokens = tokenize(input, vocab);
        auto trunc_sequence = truncate_tokens_max_len(tokens, max_len);
        auto padding_mask = create_padding_mask(trunc_sequence, max_len);

        auto output = transformer.forward(trunc_sequence, padding_mask);
        std::vector<float> pooled_output = mean_pooling(output);
        vector<float> logits = linear_layer(pooled_output, weights, bias);
        vector<float> probabilities = softmax(logits);

        cout << "Category probabilities:\n";
        for (size_t i = 0; i < categories.size(); ++i) {
            cout << categories[i] << ": " << probabilities[i] << "\n";
        }

        auto max_iter = max_element(probabilities.begin(), probabilities.end());
        size_t max_index = distance(probabilities.begin(), max_iter);
        cout << "Predicted category: " << categories[max_index] << "\n";
    }
}


int main() {
 
    bool load_parameters_yes_no = false;

    cout << "========================================================================================================" << endl;
    cout << "Transformer Test in Mini Format (C/C++) - No Use of ML Libraries" << endl;
    cout << "The goal is to build and understand the Transformer algorithm from scratch using pure C++." << endl;
    cout << "========================================================================================================" << endl;
    cout << endl;


    // Step 1: Load vocabulary
    std::unordered_map<std::string, int> vocab;
    std::string vocab_file = "vocab.txt";
    if (!load_vocab_from_file(vocab_file, vocab)) {
        std::cerr << "Failed to load vocab from: " << vocab_file << std::endl;
        return -1;
    }

    // Step 2: Load categories
    std::vector<std::string> categories;
    std::string labels_file = "labels.txt";
    std::ifstream labels_stream(labels_file);
    if (!labels_stream.is_open()) {
        std::cerr << "Error: Could not open labels file: " << labels_file << std::endl;
        return -1;
    }

    std::string line;
    while (std::getline(labels_stream, line)) {
        categories.push_back(line);
    }
    labels_stream.close();

    if (categories.empty()) {
        std::cerr << "Error: Labels file is empty." << std::endl;
        return -1;
    }

    int num_categories = categories.size();
    cout << "Loaded " << num_categories << " categories from labels.txt." << endl;

    // Step 3: Verify category files exist
    std::vector<std::string> input_files;
    for (const auto& category : categories) {
        std::string file_name = category + ".txt";
        if (!std::filesystem::exists(file_name)) {
            std::cerr << "Error: Missing file for category: " << file_name << std::endl;
            return -1;
        }
        input_files.push_back(file_name);
    }

    // Step 4: Prepare dataset
    std::vector<std::vector<int>> dataset_2D;
    std::vector<int> labels;
    std::unordered_map<std::string, int> label_map;
    for (size_t i = 0; i < categories.size(); ++i) {
        label_map[categories[i]] = i;
    }

    if (!prepare_dataset_from_files(input_files, label_map, dataset_2D, labels, vocab)) {
        std::cerr << "Failed to prepare dataset." << std::endl;
        return -1;
    }

    // ----------------------------------------------------------------
    // Then continue your existing logic...
    //   - create the Transformer
    //   - define final layer weights
    //   - run training loop, etc.
    // ----------------------------------------------------------------

    std::cout << "Do you want to load an existing model parameter with embedding matrix from a file? (Y/N, y/n): ";
    std::string choice;
    std::cin >> choice;

     if (choice == "/Y" || choice == "y")
    {
        load_parameters_yes_no = true; // Load from file
    }
    else
    {
        load_parameters_yes_no = false; // Random initialization
    }

    int length = 0;
    cout << "Tokenized Dataset:\n";
    for (size_t i = 0; i < dataset_2D.size(); ++i) {
        cout << categories[labels[i]] << ": ";
        int token_cnt = 0;
        for (int token : dataset_2D[i]) {
            cout << token << " ";
            token_cnt++;
            if (length < token_cnt) {
                length = token_cnt;
            }
        }
        cout << "\n";
    }
    cout << "Maximum token sequence length counter: " << length << endl;

    // Define parameters
    int vocab_size = 5000;
    int d_model = 64; // The "resolution" of the positional encoding and embedding space. 
                    // Think of it like a meter stick with 128 evenly spaced lines: 
                    // this determines how finely the meaning of a token can be represented
                    // across multiple dimensions.
                    //
                    // Each token (word or sub-word) is not just an isolated entity but carries 
                    // a representation that heavily depends on its position and relationships 
                    // to other tokens in the context. For example, the word "bank" could 
                    // mean "riverbank" or "financial bank," and its meaning is influenced 
                    // by neighboring words.
                    //
                    // In this context, "d_model" defines the number of dimensions (features) 
                    // used to represent these relationships. Higher d_model provides a finer 
                    // "resolution," allowing the model to encode more complex interactions 
                    // and associations across the sequence. 
                    //
                    // Increasing d_model expands the range of nuances and relationships that 
                    // the model can capture, enabling it to differentiate subtle differences 
                    // in meaning based on positional and contextual variations in the input 
                    // sequence.
                    //
                    // However, higher d_model also increases computational complexity and 
                    // the risk of overfitting for small datasets, so a balance is needed.

    int num_heads = 4;// Number of attention heads. The attention class split the Q,K and V vector bus into smaller attention vectors 
                      // and then the splitted Q_split,K_split and V_split vectors combined togheter again before enter the global Q,K and V vector bus feed forward
                      // so if num_heads = 4 and d_model = 64 each attention have only d_model/num_heads = 64/4 = 16 loacal dimentsion to calculate on
    int d_ff = 256;   // d_ff: Dimensionality of the hidden layer in the feed-forward network.
                      //       Each feed-forward network in the transformer consists of two linear layers:
                      //       - The first layer expands the input dimensionality (d_model) to a larger hidden size (d_ff).
                      //       - The second layer projects the hidden layer back down to the original dimensionality (d_model).
                      //       This expansion allows the model to learn richer, non-linear representations
                      //       by operating in a higher-dimensional space during the intermediate steps.
                      //
                      //       Typical values for d_ff are 2-4 times larger than d_model.
                      //       For example:
                      //         d_model = 128, d_ff = 256 or d_ff = 512.
                      //       This ratio balances the model's capacity with computational efficiency.
    int num_layers = 6;
   // int max_len = length; //64  Maximum sequence length (number of tokens in a single input)
    int max_len = 26;

    std::cerr << "d_model: " << d_model << std::endl;
    std::cerr << "num_heads: " << num_heads << std::endl;
    int check_num_head_settings = d_model % num_heads;
    if(check_num_head_settings != 0)
    {
        std::cerr << "Failed check_num_head_settings != 0: " << check_num_head_settings << std::endl;
        return -1;
    }


    // ================== Set up the transformer ==================
    vocab_size = vocab.size(); // Dynamically set to the actual vocabulary size
    cout << "vocab_size = " << vocab_size << endl;

    // Initialize final layer weights and bias
    std::vector<std::vector<float>> final_weights(d_model, std::vector<float>(num_categories, 0.0f));
    std::vector<float> final_bias(num_categories, 0.0f);

    if (load_parameters_yes_no) {
        if (!load_final_layer_weights(final_weights, final_bias)) {
            // Fall back to random initialization if loading fails
            std::srand(std::time(0));
            for (auto& row : final_weights) {
                for (auto& val : row) {
                    val = static_cast<float>(std::rand()) / RAND_MAX; // Random values between 0 and 1
                }
            }
            for (auto& val : final_bias) {
                val = static_cast<float>(std::rand()) / RAND_MAX;
            }
        }
    } else {
        // Random initialization
        std::srand(std::time(0));
        for (auto& row : final_weights) {
            for (auto& val : row) {
                val = static_cast<float>(std::rand()) / RAND_MAX;
            }
        }
        for (auto& val : final_bias) {
            val = static_cast<float>(std::rand()) / RAND_MAX;
        }
    }

    // Create transformer
    Transformer transformer(vocab_size, d_model, max_len, num_heads, d_ff, num_layers, load_parameters_yes_no);

    cout << "Do you want to start mini prompt mode? (Y/N): ";
    string response;
    cin >> response;
    cin.ignore(); // Ignore trailing newline character from cin

    if (response == "Y" || response == "y" || response == "Yes" || response == "yes" || response == "YES") {
        run_prompt_mode(transformer, max_len, vocab, final_weights, final_bias, categories);
    } else {
        cout << "Continuing with training loop...\n";
        
    // ============== Training loop ===================
    int epochs = 1000;
    // Initialize velocity for weights and bias
    std::vector<std::vector<float>> velocity_weights(final_weights.size(),
                                                     std::vector<float>(final_weights[0].size(), 0.0f));
    std::vector<float> velocity_bias(final_bias.size(), 0.0f);

    GLOBAL_learning_rate = 0.0001;//0.0001
    GLOBAL_momentum = 0.9;
    GLOBAL_ATTENTION_learning_rate = GLOBAL_learning_rate *1.0;//0.1
    GLOBAL_ATTENTION_momentum = GLOBAL_momentum*1.0; //0.5  
    std::cout << "learning_rate: " << GLOBAL_learning_rate << std::endl;
    std::cout << "momentum: " << GLOBAL_momentum << std::endl;
    // Training loop with gradient computation
    float best_avg_loss = 10000.0;
    const int print_dot_interval = 10;
    int print_dot_cnt = 0;
    for (int epoch = 1; epoch <= epochs; ++epoch)
    {


        std::cout << "Epoch " << epoch << " / " << epochs << "\n";
        // Shuffle dataset
        fisher_yates_shuffle(dataset_2D, labels);
        float epoch_loss = 0.0f; // Accumulate loss for the epoch
        int correct_prob_cnt = 0;
        int data_set_cnt = 0;
        for (size_t i = 0; i < dataset_2D.size(); ++i)
        {
            if(print_dot_cnt < print_dot_interval)
            {
                print_dot_cnt++;
            }
            else
            {
                cout << "." << flush; 
                print_dot_cnt = 0;
            }
            // Prepare input and padding mask
           // auto padded_input = pad_sequence(dataset_2D[i], max_len);
            auto trunc_sequence = truncate_tokens_max_len(dataset_2D[i], max_len);
            auto padding_mask = create_padding_mask(trunc_sequence, max_len);

            // Forward pass through transformer
            auto output_trans = transformer.forward(trunc_sequence, padding_mask);

            // Reduce transformer output (e.g., by mean pooling)
            std::vector<float> pooled_output = mean_pooling(output_trans);

            // Apply final classification layer
            std::vector<float> logits = linear_layer(pooled_output, final_weights, final_bias);
            std::vector<float> probabilities = softmax(logits);
         //   cout << "Size of probabilities: " << probabilities.size() << endl;
            int idx=0;
            int predicted_idx = 0;
            float predict_max = 0.0;
            for(auto val : probabilities)
            {
                if(predict_max < val)
                {
                    predict_max = val;
                    predicted_idx = idx; 
                }
           //     cout << "probabilities[ " << idx << "] : "<< val << endl;
                idx++;
            }
           // cout << " labels[" << i << "] : " << labels[i] << " predicted_idx : " << predicted_idx << endl;
            if(predicted_idx == labels[i])
            {
                correct_prob_cnt++;
            }
            // Backpropagation starts here
            // Step 1: Compute gradient of loss with respect to logits
            std::vector<float> grad_logits = cross_entropy_loss_gradient(probabilities, labels[i]);
            // Step 2: Compute gradients for final weights and bias
            std::vector<std::vector<float>> grad_final_weights(final_weights.size(),
                                                               std::vector<float>(final_weights[0].size(), 0.0f));
            std::vector<float> grad_final_bias(final_bias.size(), 0.0f);

            for (size_t j = 0; j < pooled_output.size(); ++j)
            {
                for (size_t k = 0; k < grad_logits.size(); ++k)
                {
                    grad_final_weights[j][k] += pooled_output[j] * grad_logits[k];
                }
            }

            for (size_t k = 0; k < grad_logits.size(); ++k)
            {
                grad_final_bias[k] += grad_logits[k];
            }

            // Step 3: Update final weights and bias using SGD with momentum
            for (size_t j = 0; j < final_weights.size(); ++j)
            {
                for (size_t k = 0; k < final_weights[0].size(); ++k)
                {
                    velocity_weights[j][k] = GLOBAL_momentum * velocity_weights[j][k] - GLOBAL_learning_rate * grad_final_weights[j][k];
                    final_weights[j][k] += velocity_weights[j][k];
                }
            }

            for (size_t k = 0; k < final_bias.size(); ++k)
            {
                velocity_bias[k] = GLOBAL_momentum * velocity_bias[k] - GLOBAL_learning_rate * grad_final_bias[k];
                final_bias[k] += velocity_bias[k];
            }

            // Compute gradient of final layer with respect to mean pooling output
            std::vector<float> pooled_output_gradient(pooled_output.size(), 0.0f);
            for (size_t i = 0; i < final_weights.size(); ++i) {
                for (size_t j = 0; j < grad_logits.size(); ++j) {
                    pooled_output_gradient[i] += grad_logits[j] * final_weights[i][j];
                }
            }

            // Backpropagate through mean pooling
            std::vector<std::vector<float>> grad_pooled = mean_pooling_backward(output_trans, pooled_output_gradient);


            // Backpropagate gradient through the Transformer 
           transformer.backward(grad_pooled);

           // print_out_probabilities(probabilities, padded_input);// Print probabilities for debugging
           //  Compute loss and accumulate
           float loss = cross_entropy_loss(probabilities, labels[i]);
           epoch_loss += loss;

          data_set_cnt++;
        }
        float correct_prob = (float)correct_prob_cnt/(float)data_set_cnt;
        cout << "** correct_prob : " << correct_prob << endl;
        float avg_loss_this_epoch = epoch_loss / dataset_2D.size();
        if(best_avg_loss > avg_loss_this_epoch)
        {
            best_avg_loss = avg_loss_this_epoch;
            save_final_layer_weights(final_weights, final_bias);
            transformer.save_embedding_matrix();
            transformer.save_attention_weights();
            transformer.save_feed_forward_weights();
            transformer.save_LayerNormalization_weights();
        }
           
        // Print average loss for the epoch
        std::cout << "Average Loss for Epoch " << epoch << ": " << (epoch_loss / dataset_2D.size()) << "\n";
    }
    //========================== End training loop ===================

    }

    // Save final layer weights (optional)
    save_final_layer_weights(final_weights, final_bias);
    transformer.save_embedding_matrix();
    transformer.save_attention_weights();
    transformer.save_feed_forward_weights();    
    transformer.save_LayerNormalization_weights();
    return 0;

}
