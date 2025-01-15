#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <algorithm>
#include <set>
#include <cctype> // Make sure to include this for tolower

using namespace std;

// Function to clean a word by removing only ? and . and converting to lowercase
string cleanWord(string word) {
    string cleanedWord = "";
    for (char c : word) {
        if (c != '?' && c != '.') {
            cleanedWord += tolower(c);
        }
    }
    return cleanedWord;
}

int main() {
    ifstream inputFile("input.txt");
    ofstream outputFile("vocab_extracted.txt");

    if (!inputFile.is_open()) {
        cerr << "Error opening input file." << endl;
        return 1;
    }

    if (!outputFile.is_open()) {
        cerr << "Error opening output file." << endl;
        return 1;
    }

    set<string> uniqueWords; // Use a set to store unique words

    // Write the special tokens at the beginning
    outputFile << "[PAD]" << endl;
    outputFile << "[UNK]" << endl;

    uniqueWords.insert("[PAD]");
    uniqueWords.insert("[UNK]");

    string line;
    while (getline(inputFile, line)) {
        stringstream ss(line);
        string word;
        while (ss >> word) {
            string cleanedWord = cleanWord(word);

            // Check if the cleaned word is one of the reserved words (case-insensitive)
            if (!cleanedWord.empty() && 
                cleanedWord != "[pad]" && cleanedWord != "[unk]" &&
                cleanedWord != "[PAD]" && cleanedWord != "[UNK]" &&
                uniqueWords.find(cleanedWord) == uniqueWords.end()) {

                uniqueWords.insert(cleanedWord);
                outputFile << cleanedWord << endl; // New line after each word
            }
        }
    }

    inputFile.close();
    outputFile.close();

    cout << "Vocabulary extraction complete. Output written to vocab_extracted.txt" << endl;

    return 0;
}