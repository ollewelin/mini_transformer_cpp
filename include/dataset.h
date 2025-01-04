#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>

// Tokenize a sentence into a vector of integers based on the vocabulary
std::vector<int> tokenize(const std::string &sentence, const std::unordered_map<std::string, int> &vocab);

// Prepare the dataset: tokenized sentences and labels
void prepare_dataset(std::vector<std::vector<int>> &data, std::vector<int> &labels, const std::unordered_map<std::string, int> &vocab);

#endif // DATASET_H
