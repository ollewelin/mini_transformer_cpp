#include "dataset.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cctype>
#include <string>

// This function tokenizes a sentence using the vocabulary
std::vector<int> tokenize(const std::string &sentence, const std::unordered_map<std::string, int> &vocab) {
    std::istringstream iss(sentence);
    std::string word;
    std::vector<int> tokens;

    while (iss >> word) {
        // Convert word to lowercase for consistency
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        // Remove punctuation from the word
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());

        // If word not found, use [UNK] token. We assume [UNK] is ID=1 in the vocab.
        if (vocab.count(word)) {
            tokens.push_back(vocab.at(word));
        } else {
            tokens.push_back(vocab.at("[UNK]")); 
        }
    }

    return tokens;
}

// ------------- NEW: load vocab from file -------------
// Assumes each line contains exactly one token, e.g.
// [PAD]
// [UNK]
// what
// time
// is
bool load_vocab_from_file(const std::string &vocab_file, std::unordered_map<std::string, int> &vocab)
{
    std::ifstream ifs(vocab_file);
    if (!ifs.is_open()) {
        std::cerr << "Error: Could not open vocab file: " << vocab_file << std::endl;
        return false;
    }

    vocab.clear();  // Clear existing vocab
    std::string token;
    int index = 0;

    while (std::getline(ifs, token)) {
        // Remove extra whitespace
        token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
        // Skip empty lines
        if (token.empty()) continue;
        
        // Insert into vocab
        vocab[token] = index;
        index++;
    }

    ifs.close();

    if (vocab.empty()) {
        std::cerr << "Warning: The vocabulary file is empty or not loaded properly.\n";
        return false;
    }

    std::cout << "Loaded vocabulary of size: " << vocab.size() << std::endl;
    return true;
}

// ------------- NEW: prepare dataset from question and answer files -------------
bool prepare_dataset_from_files(const std::string &question_file,
                                const std::string &answer_file,
                                std::vector<std::vector<int>> &data,
                                std::vector<int> &labels,
                                const std::unordered_map<std::string, int> &vocab)
{
    std::ifstream ifs_q(question_file);
    std::ifstream ifs_a(answer_file);

    if (!ifs_q.is_open()) {
        std::cerr << "Error: Could not open question file: " << question_file << std::endl;
        return false;
    }
    if (!ifs_a.is_open()) {
        std::cerr << "Error: Could not open answer file: " << answer_file << std::endl;
        return false;
    }

    data.clear();
    labels.clear();

    // Read questions, label=0
    std::string line;
    while (std::getline(ifs_q, line)) {
        if (line.empty()) continue;  // skip empty lines
        std::vector<int> question_tokens = tokenize(line, vocab);
        data.push_back(question_tokens);
        labels.push_back(0); // 0 for Question
    }

    // Read answers, label=1
    while (std::getline(ifs_a, line)) {
        if (line.empty()) continue; // skip empty lines
        std::vector<int> answer_tokens = tokenize(line, vocab);
        data.push_back(answer_tokens);
        labels.push_back(1); // 1 for Answer
    }

    ifs_q.close();
    ifs_a.close();

    // Quick check
    std::cout << "Loaded " << data.size() << " examples total (Questions + Answers)." << std::endl;
    return true;
}
