#include "dataset.h"
#include <iostream>

// Tokenize a sentence using the vocabulary
std::vector<int> tokenize(const std::string &sentence, const std::unordered_map<std::string, int> &vocab) {
    std::istringstream iss(sentence);
    std::string word;
    std::vector<int> tokens;

    while (iss >> word) {
        // Convert word to lowercase for consistency
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);

        // Remove punctuation from the word
        word.erase(std::remove_if(word.begin(), word.end(), ispunct), word.end());

        // Add token or unknown token (0)
        tokens.push_back(vocab.count(word) ? vocab.at(word) : 0);
    }

    return tokens;
}

// Prepare the dataset
void prepare_dataset(std::vector<std::vector<int>> &data, std::vector<int> &labels, const std::unordered_map<std::string, int> &vocab) {
    // Questions
    std::vector<std::string> questions = {
        "What time is it now?",
        "How are you doing today?",
        "Can you help me with this?",
        "Where is the nearest bus stop?",
        "Why is the sky blue?",
        "Who wrote this book?",
        "Which movie do you recommend?",
        "When will the meeting start?",
        "Is it going to rain today?",
        "Could you explain that again?"
    };

    // Answers
    std::vector<std::string> answers = {
        "It is three o'clock.",
        "I am doing well, thank you.",
        "Yes, I can help you with that.",
        "The nearest bus stop is down the street.",
        "The sky appears blue because of light scattering.",
        "This book was written by Jane Austen.",
        "I recommend watching 'Inception.'",
        "The meeting will start in ten minutes.",
        "Yes, it is going to rain later today.",
        "Sure, I'll explain it again."
    };

    // Labels: 0 for Questions, 1 for Answers
    for (const auto &q : questions) {
        data.push_back(tokenize(q, vocab));
        labels.push_back(0); // Question
    }

    for (const auto &a : answers) {
        data.push_back(tokenize(a, vocab));
        labels.push_back(1); // Answer
    }
}
