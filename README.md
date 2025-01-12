# Tutorial code for C/C++ Transformer "mini_transformer_cpp"

I have done lot's of tweeks to focus on understand this transformer algorithm in C/C++ perspective and also generate print outs and graphs down the road to understand the code and the transformer algorithm.

### Overview and purpose:
Transformer Test in Mini Format (C/C++) - No Use of ML Libraries. Just for my own tutorial proposal in the first hand but may enjoy other as well.

## mini prompt test if Question or Answer

![](mini_prompt.jpeg)

### Description of the project:
Transformer Test in Mini Format (C/C++) - No Use of ML Libraries. The goal is to build and understand the Transformer algorithm from scratch using pure C++. Testing key components step by step...



## File structures
```
transformer_project/
├── src/
│   ├── main.cpp            # Entry point for the application
│   ├── config.cpp
│   ├── dataset.cpp         # Mini dataset toy example
│   ├── layer_normalization.cpp
│   ├── transformer.cpp     # Implementation of the Transformer
│   ├── attention.cpp       # Implementation of the attention mechanism
│   ├── feed_forward.cpp    # Implementation of the feed-forward network
│   ├── embedding.cpp       # Implementation of embeddings
│   ├── positional_encoding.cpp # Implementation of positional encoding
│   └── utils.cpp           # Helper functions
├── include/
│   ├── transformer.h
│   ├── config.h            # Contain preprocessor directives (Global preprocessor directives) and Global variables 
│   ├── attention.h
│   ├── dataset.h
│   ├── layer_normalization.h
│   ├── feed_forward.h
│   ├── embedding.h
│   ├── positional_encoding.h
│   └── utils.h
├── build/                  # Directory for compiled files
├── data/                   # Sample input data for testing
└── Makefile                # Build instructions
```
## Understand position encoder
Code:

        positional_encoding.h
        positional_encoding.cpp
        

Purpose of Positional Encoding

The purpose of the positional encoding is to inject information about the position of each token in the sequence. Transformers process sequences in parallel without inherent knowledge of token order, so positional encoding compensates for this.

Key Idea:

        pos_encoding[pos][i] 
        
stores the positional encoding for the i-th dimension of the pos-th token.
Sine (sin) and Cosine (cos) functions with varying frequencies are used to encode positions.


### "d_model" is like set the resolution of dimentions of distance measure 

```
    int d_model = 128; // The "resolution" of the positional encoding and embedding space. 
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
```

## Position encoder initialized vector values

Example printout of sin
```
for (int pos = 0; pos < max_len; ++pos)
    {
        for (int i = 0; i < d_model; ++i)
        {
        ....
pos = tokens position
i = dimentions
pos_encoding[pos][i]

example pos 0 and 1
sin pos_encoding[0][122]: 0
sin pos_encoding[0][124]: 0
sin pos_encoding[0][126]: 0
sin pos_encoding[1][0]: 0.841471
sin pos_encoding[1][2]: 0.76172
sin pos_encoding[1][4]: 0.681561
```

I was plot the initialized values of pos_encoding[pos][i] to show a plot what the contructor doing.
### Plot the sin initilized vector
Use the octave to plot sin.dat file
example data:
```
0 0 0
0 2 0
0 4 0
0 6 0
...
0 122 0
0 124 0
0 126 0
1 0 0.841471
1 2 0.76172
1 4 0.681561
...
1 122 0.000153993
1 124 0.000133352
1 126 0.000115478
2 0 0.909297
2 2 0.987046
2 4 0.99748
```

#### all dimentions sin
![](build/sin_position_at_constructor.png)
#### sin dim 0 2 4
![](build/sin_pos_dim_0_2_4_only.png)

### Plot the cos initilized vector
Use the octave to plot cos.dat file
example data:
```
0 1 1
0 3 1
0 5 1
...
0 125 1
0 127 1
1 1 0.540302
1 3 0.647906
...
1 123 1
1 125 1
1 127 1
2 1 -0.416147
2 3 -0.160436
2 5 0.0709483
```
#### all dimentions cos
![](build/cos_position_at_constructor.png)
#### cos dim 1 3 5
![](build/cos_pos_dim_1_3_5_only.png)


## Embedding_matrix
Code: 

        embedding.h
        embedding.cpp

### 1. Purpose of embedding_matrix

The embedding_matrix serves as a lookup table for mapping each token ID (integer) to a high-dimensional vector of size d_model.

vocab_size (Rows):
Each row corresponds to a unique token in the vocabulary.
Example: If "The" has token ID 42, row 42 of the embedding matrix contains the vector representation of "The".

d_model (Columns):
Each column in the row represents one dimension of the token's embedding.
Example: If d_model = 128, each token is represented as a 128-dimensional vector.

### 2. What the Code Does

        embedding_matrix = std::vector<std::vector<float>>(vocab_size, std::vector<float>(d_model));

This code:
Creates a 2D vector:
The outer vector has vocab_size rows (5000 in your case).
Each row is an inner vector of size d_model (128 in your example).

Initializes All Values to Zero:
By default, all elements in the matrix are initialized to 0.0.

### Example

#### 1. Let’s assume:

            vocab_size = 3 (3 tokens in the vocabulary).
            d_model = 4 (4-dimensional embeddings).
#### 2. Initial matrix:

```
embedding_matrix = std::vector<std::vector<float>>(3, std::vector<float>(4));

embedding_matrix = [
    [0.0, 0.0, 0.0, 0.0],  // Token 0
    [0.0, 0.0, 0.0, 0.0],  // Token 1
    [0.0, 0.0, 0.0, 0.0]   // Token 2
];
```
#### 3. Can Be Updated with Random or Learned Values:

Later, this matrix is either:
Initialized with random values (during training).
Filled with pre-trained embeddings (e.g., Word2Vec or GloVe).


Suppose we update it with random values:
```
embedding_matrix = [
    [0.12, 0.34, -0.56, 0.78],  // Token 0
    [0.91, -0.45, 0.67, -0.23], // Token 1
    [0.05, 0.14, -0.32, 0.89]   // Token 2
];

```
  
#### 4. How It Works During a Forward Pass

4.1 Input Tokens (IDs):
A sentence like "The cat sat" is tokenized into IDs (e.g., [42, 18, 87]).

4.2 Lookup in the Embedding Matrix:
For each token ID, the corresponding row in embedding_matrix is retrieved:
```
Token ID 42 → Row 42 (vector for "The").
Token ID 18 → Row 18 (vector for "cat").
Token ID 87 → Row 87 (vector for "sat").
```

4.3 Result:
A 2D vector is created where:
Rows correspond to tokens in the input sequence.
Columns correspond to the embedding dimensions (d_model).

### Summary

        vocab_size 

defines the number of rows in the matrix, representing the number of unique tokens the model can understand.
        
        d_model 

defines the number of columns, representing the size of each token's embedding vector.
The matrix acts as a lookup table, converting token IDs into dense vector representations.

## attention class
The center of the tranformer seem to be the attention block for in my view. It's lot of matrix multiplication used here.

I was made a print out test of one single attention layer with one single attention head.
You can "un-comment" the line

        #define PRINT_OUT_TEST_ATTENTION_FORWARD_OPERATION

at

        config.h

To enable this test print out of attention mechanism:

```
==================== Test: Single Attention Layer ====================
Matrix sizes are valid.
The resolution of the positional encoding and embedding space, d_model: 4
Attention weights for layer 0 initialized with random values.
Randomized attention weights for layer 0 saved to files.

=== Relationship Between d_model, num_heads, and Matrix Dimensions ===
d_model (total embedding dimension): 4
num_heads (number of attention heads): 1
d_k (key/query dimension per head): 4
d_v (value dimension per head): 4

Explanation:
- The total embedding dimension (d_model) is divided among all attention heads.
- With num_heads = 1, each head gets the full d_model, so d_k = d_model / num_heads = 4.
- Similarly, d_v = d_model / num_heads = 4.
In this case, each token is represented with 4 dimensions in Q and K, and 4 dimensions in V.

=== Hard coded Test Input Matrices ===

Input Q (Query):
[
  [1.0000, 0.5000, 0.1000, 0.0100]
  [0.2000, 1.3000, 0.2000, 0.0200]
  [1.2000, 2.3000, 3.2000, 4.1100]
]
Each row represents a token, and each column represents one of the 4 dimensions of the query vector.

Input K (Key):
[
  [0.8000, 0.3000, 0.3000, 0.0300]
  [0.1000, 0.9000, 0.4000, 0.0400]
  [0.2000, 0.3000, 3.0000, 1.1100]
]
Each row represents a token, and each column represents one of the 4 dimensions of the key vector.

Input V (Value):
[
  [1.2000, 0.7000, 0.5000, 0.0500]
  [0.5000, 0.4000, 0.6000, 0.0600]
  [2.2000, 1.3000, 0.0000, 3.1100]
]
Each row represents a token, and each column represents one of the 4 dimensions of the value vector.

Summary:
- Q and K have 4 columns because they encode positional and content-related similarities.
- V has 4 columns because it contains the actual token content to be weighted and combined.
=====================================================================

=== Scaled Dot-Product Attention Debug Output ===

Step 1: Compute QK^T
Query (Q) matrix (shape: 3 x 4):
[
  [1.0000, 0.5000, 0.1000, 0.0100]
  [0.2000, 1.3000, 0.2000, 0.0200]
  [1.2000, 2.3000, 3.2000, 4.1100]
]
Key (K) matrix (shape: 3 x 4):
[
  [0.8000, 0.3000, 0.3000, 0.0300]
  [0.1000, 0.9000, 0.4000, 0.0400]
  [0.2000, 0.3000, 3.0000, 1.1100]
]
QK^T (scores matrix, shape: 3 x 3):
[
  [0.9803, 0.5904, 0.6611]
  [0.6106, 1.2708, 1.0522]
  [2.7333, 3.6344, 15.0921]
]
Each element in this matrix represents the dot product similarity between a query vector (row) and a key vector (column).
For example:
  - scores[0][0] = dot product of Q[0] and K[0] (similarity between token 1's query and token 1's key).
  - scores[0][1] = dot product of Q[0] and K[1] (similarity between token 1's query and token 2's key).
  - scores[1][2] = dot product of Q[1] and K[2] (similarity between token 2's query and token 3's key).
Each row represents the similarity of a specific token's query with all tokens' keys, and each column represents the similarity of all queries with a specific token's key.

Step 2: Scale scores by sqrt(d_k)
Scaling factor (sqrt(d_k)): 2.0000
Scaled scores matrix:
[
  [0.4902, 0.2952, 0.3306]
  [0.3053, 0.6354, 0.5261]
  [1.3667, 1.8172, 7.5461]
]
Each score is scaled to adjust for the dimensionality of the key vectors.

Step 3: Apply masking to prevent attending to future tokens
Masked scores matrix:
[
  [0.4902, -inf, -inf]
  [0.3053, 0.6354, -inf]
  [1.3667, 1.8172, 7.5461]
]
This matrix shows the scores after applying a mask to ensure that a token only attends to itself and earlier tokens.

Step 4: Apply softmax to scores
Softmax applied (attention weights):
[
  [1.0000, 0.0000, 0.0000]
  [0.4182, 0.5818, 0.0000]
  [0.0021, 0.0032, 0.9947]
]
Each row represents the attention distribution for a token. The values sum to 1, showing how much each token attends to other tokens.

Step 5: Multiply scores with Value (V) matrix
Value (V) matrix (shape: 3 x 4):
[
  [1.2000, 0.7000, 0.5000, 0.0500]
  [0.5000, 0.4000, 0.6000, 0.0600]
  [2.2000, 1.3000, 0.0000, 3.1100]
]
Output matrix (shape: 3 x 4):
[
  [1.2000, 0.7000, 0.5000, 0.0500]
  [0.7928, 0.5255, 0.5582, 0.0558]
  [2.1924, 1.2959, 0.0030, 3.0938]
]
Each row in the output matrix corresponds to the weighted sum of value vectors for each token, based on its attention distribution.
=== End of Debug Output ===
=====================================================================

```
## Example log when training 

```

Average Loss for Epoch 258: 0.354806
Epoch 259 / 1000
** correct_prob : 0.855
Average Loss for Epoch 259: 0.352544
Epoch 260 / 1000
** correct_prob : 0.855
Average Loss for Epoch 260: 0.356734
Epoch 261 / 1000
** correct_prob : 0.855
Average Loss for Epoch 261: 0.356308
Epoch 262 / 1000
** correct_prob : 0.855
Average Loss for Epoch 262: 0.350175
Epoch 263 / 1000
** correct_prob : 0.8625
Final layer weights saved to final_layer_weight.bin.
Embedding matrix saved to file: embedding_matrix.bin
Attention weights for layer 0 saved to files.
Attention weights for layer 1 saved to files.
Attention weights for layer 2 saved to files.
Attention weights for layer 3 saved to files.
Attention weights for layer 4 saved to files.
Attention weights for layer 5 saved to files.
FeedForward weights for layer 0 initialized and saved to file.
FeedForward weights for layer 1 initialized and saved to file.
FeedForward weights for layer 2 initialized and saved to file.
FeedForward weights for layer 3 initialized and saved to file.
FeedForward weights for layer 4 initialized and saved to file.
FeedForward weights for layer 5 initialized and saved to file.
Average Loss for Epoch 263: 0.330319
Epoch 264 / 1000
** correct_prob : 0.8775
Average Loss for Epoch 264: 0.350765
Epoch 265 / 1000
** correct_prob : 0.86
Average Loss for Epoch 265: 0.369768
Epoch 266 / 1000
** correct_prob : 0.865
Average Loss for Epoch 266: 0.362699
Epoch 267 / 1000
** correct_prob : 0.8775
Average Loss for Epoch 267: 0.342921
Epoch 268 / 1000
** correct_prob : 0.8875
Average Loss for Epoch 268: 0.335516
Epoch 269 / 1000
** correct_prob : 0.8725
Average Loss for Epoch 269: 0.344732

```

## Example start program and run mini prompt


```

./transformer_app 
========================================================================================================
Transformer Test in Mini Format (C/C++) - No Use of ML Libraries
The goal is to build and understand the Transformer algorithm from scratch using pure C++.
========================================================================================================

Loaded vocabulary of size: 1226 from vocab.txt
Loaded 400 examples total (Questions + Answers).
Do you want to load an existing model parameter with embedding matrix from a file? (Y/N, y/n): y
Tokenized Dataset:
Question: 1187 1091 521 523 668 
Question: 475 73 1223 289 1094 
Question: 140 1223 462 17 1202 1083 1058 
Question: 1190 521 1071 650 134 1019 
Question: 1195 521 1071 978 117 
Question: 1194 17 1083 78 
Question: 1191 17 285 1223 833 
Question: 1189 1199 1071 17 1013 
Question: 521 523 440 1093 820 1096 
Question: 17 1223 352 1070 43 
Question: 1187 521 1071 144 675 416 
Question: 475 140 480 491 646 171 17 
Question: 140 1223 959 17 54 340 675 1070 423 
Question: 1190 140 480 384 638 875 
Question: 1195 288 1071 17 17 492 1071 17 
Question: 1194 268 729 
Question: 1191 793 537 958 480 549 388 
Question: 1189 288 1071 1021 687 
Question: 521 523 909 1093 1117 21 668 
Question: 17 1223 752 858 1071 814 
Question: 1187 521 1224 691 682 1071 230 17 
Question: 475 285 480 774 403 18 527 512 
Question: 140 1223 197 1071 252 675 1071 869 
Question: 1190 521 1071 107 748 1093 17 75 464 
Question: 1195 285 1178 652 1093 1146 756 492 17 
Question: 1194 140 462 17 1202 1083 795 
Question: 1191 557 285 480 652 1093 502 
Question: 1189 521 1071 657 134 77 
Question: 521 809 108 1068 525 
Question: 17 1223 968 1070 328 
Question: 1187 521 1071 606 675 559 
Question: 475 17 285 1223 344 
Question: 140 1223 83 17 1202 646 17 
Question: 1190 958 480 713 646 145 
Question: 1195 73 1077 985 594 1012 492 1071 978 
Question: 1194 17 1071 1208 17 541 1218 
Question: 1191 17 285 1223 833 403 1062 656 
Question: 1189 521 1071 240 403 1071 1030 
Question: 521 1083 1071 220 1177 1093 17 1071 1058 
Question: 17 1223 752 995 638 17 
Question: 1187 521 1071 17 132 492 1071 1208 
Question: 475 285 480 871 646 722 
Question: 140 1223 352 1071 192 675 498 492 17 
Question: 1190 958 1178 471 1071 194 
Question: 1195 521 1071 236 665 17 17 
Question: 1194 140 17 1083 806 
Question: 1191 415 285 1223 384 17 1093 1146 
Question: 1189 73 1223 750 1093 1056 18 17 
Question: 521 1070 18 1152 74 
Question: 17 1223 17 1071 1017 1093 17 1071 130 
Question: 1187 521 1071 602 1000 17 682 1083 897 
Question: 475 285 480 226 18 655 435 124 
Question: 140 1223 797 1083 286 403 17 
Question: 1190 261 1223 549 20 585 550 
Question: 1195 521 646 17 17 18 933 372 
Question: 1194 140 72 646 551 864 
Question: 1191 1181 285 1223 1146 403 394 121 
Question: 1189 1199 1071 855 99 17 
Question: 521 165 17 17 403 17 
Question: 17 1223 159 1071 402 976 752 
Question: 1187 521 1071 107 70 1093 17 1083 786 
Question: 475 261 1223 433 17 492 793 
Question: 140 1223 1114 1083 941 403 17 
Question: 1190 285 480 1031 646 349 861 
Question: 1195 288 646 739 98 295 985 816 
Question: 1194 521 1071 587 202 736 403 1046 
Question: 1191 17 285 1223 1081 521 638 308 
Question: 1189 958 1178 916 1071 657 17 
Question: 521 172 694 17 108 403 791 
Question: 17 1223 953 1071 776 17 
Question: 1187 521 1071 540 17 682 303 
Question: 475 285 480 692 1083 1007 812 
Question: 140 1223 462 17 17 1078 186 330 
Question: 1190 140 480 835 1078 17 
Question: 1195 285 586 460 17 
Question: 1194 521 1224 373 17 17 
Question: 1191 125 675 17 17 1223 833 
Question: 1189 521 1071 107 1091 1093 1169 17 
Question: 521 523 734 1093 127 17 17 
Question: 17 1223 486 1071 192 1202 54 340 
Question: 1187 521 1224 1024 403 17 
Question: 475 285 480 159 1071 1053 951 
Question: 140 1223 411 17 1071 312 17 
Question: 1190 285 1079 473 1071 17 194 
Question: 1195 521 1070 620 248 
Question: 1194 521 17 403 964 1071 207 
Question: 1191 739 634 456 1071 107 139 
Question: 1189 261 1071 17 388 687 
Question: 521 47 1071 236 17 17 17 
Question: 17 1223 17 1224 541 17 
Question: 1187 521 1071 107 724 1093 17 646 147 
Question: 475 288 684 101 18 108 804 17 
Question: 140 1223 1160 1071 1104 221 
Question: 1190 140 480 549 638 20 811 191 
Question: 1195 73 17 375 17 137 242 
Question: 1194 521 440 1093 222 1071 1117 17 
Question: 1191 1066 17 460 1178 665 17 1221 
Question: 1189 521 1071 795 348 1093 544 
Question: 521 535 675 17 332 403 1083 899 
Question: 17 1223 161 1071 577 403 330 
Question: 1187 521 1224 373 793 537 
Question: 475 285 480 502 1083 17 682 565 
Question: 140 1223 17 1071 861 137 1096 
Question: 1190 261 1223 930 1071 632 381 541 
Question: 1195 285 988 730 772 850 1205 
Question: 1194 140 613 655 1061 17 
Question: 1191 175 17 285 1223 17 107 
Question: 1189 1199 1223 847 1071 657 1161 
Question: 521 1083 18 130 694 17 17 
Question: 17 1223 452 17 1088 1071 196 
Question: 1187 521 1071 17 109 1060 58 1128 
Question: 475 285 480 33 18 655 1148 1093 1071 1053 
Question: 140 1223 249 1224 1206 403 17 17 
Question: 1190 285 480 384 1071 678 287 
Question: 1195 521 1071 705 17 492 17 
Question: 1194 521 88 1093 24 1071 946 17 
Question: 1191 862 17 1071 543 992 17 
Question: 1189 521 1071 987 1141 17 
Question: 521 1077 18 17 92 403 17 
Question: 17 1223 1044 1071 587 17 
Question: 1187 521 1071 17 233 17 17 
Question: 475 285 480 213 1083 380 1093 727 
Question: 140 1223 226 18 95 675 1071 237 
Question: 1190 140 480 930 1071 1053 577 
Question: 1195 285 988 17 624 578 17 
Question: 1194 1176 1071 388 736 1093 17 17 338 
Question: 1191 17 987 285 1223 772 
Question: 1189 285 1178 17 1093 17 94 419 1071 168 
Question: 521 1077 18 17 1206 403 17 17 
Question: 17 1223 247 475 1093 1146 1083 17 
Question: 1187 521 1071 540 17 492 697 17 1053 
Question: 475 285 480 189 18 400 513 18 1225 380 
Question: 140 1223 916 18 17 138 403 657 1182 
Question: 1190 521 1071 107 1006 1093 17 1071 17 
Question: 1195 288 612 1145 1003 300 17 
Question: 1194 958 1178 516 1093 1071 17 948 
Question: 1191 70 17 1093 108 732 
Question: 1189 285 1223 1081 1178 140 1209 17 1083 795 
Question: 521 523 17 17 1071 17 17 
Question: 17 1223 197 1071 17 35 
Question: 1187 521 1071 17 17 1091 403 1083 256 
Question: 475 285 480 198 1093 1071 1173 
Question: 140 1223 699 1071 1059 403 1083 17 
Question: 1190 140 480 828 20 1224 17 1154 
Question: 1195 285 1178 652 1093 962 18 17 17 
Question: 1194 140 17 18 17 492 435 
Question: 1191 392 675 565 285 1223 833 403 17 
Question: 1189 140 1178 17 1071 382 250 17 
Question: 521 1070 1071 220 1145 675 1071 66 
Question: 17 1223 17 1071 786 492 967 17 
Question: 1187 521 1071 107 417 874 1093 549 236 17 
Question: 475 140 480 245 646 25 17 
Question: 140 1223 411 17 1071 1061 202 566 
Question: 1190 521 1071 655 677 572 
Question: 1195 521 1071 287 494 
Question: 1194 521 1071 587 1148 675 1083 374 
Question: 1191 56 17 285 1178 1146 
Question: 1189 288 1071 17 17 
Question: 521 1224 1053 185 1202 17 
Question: 17 1223 17 17 675 697 929 759 
Question: 1187 521 1071 107 1177 1093 70 17 
Question: 475 285 480 852 18 435 181 
Question: 140 1223 687 18 1046 1090 403 17 
Question: 1190 140 480 294 1071 17 923 
Question: 1195 521 1071 787 17 985 578 
Question: 1194 958 480 35 492 1071 222 17 
Question: 1191 482 285 640 17 464 1146 
Question: 1189 285 1223 749 682 17 1071 723 
Question: 521 523 17 85 1224 573 668 
Question: 17 1223 462 17 250 1071 1148 17 
Question: 1187 521 1071 107 17 1093 433 1093 1071 17 
Question: 475 285 480 836 225 1145 
Question: 140 1223 247 475 1093 905 1078 17 
Question: 1190 140 480 930 1071 1148 376 577 
Question: 1195 73 1178 1109 1078 621 
Question: 1194 140 1056 17 675 1071 655 374 
Question: 1191 325 17 285 480 652 1093 949 
Question: 1189 521 1071 657 17 17 17 
Question: 521 646 236 909 1202 1083 196 
Question: 17 1223 126 17 1071 221 17 
Question: 1187 521 1071 107 771 403 722 1020 
Question: 475 285 480 1141 646 792 497 
Question: 140 1223 528 1071 1165 194 492 390 17 
Question: 1190 285 1178 1021 17 17 
Question: 1195 521 1071 17 223 1018 584 
Question: 1194 521 440 1093 588 1071 17 17 
Question: 1191 203 751 73 1178 1150 
Question: 1189 73 1178 17 1071 17 124 
Question: 521 1077 18 17 403 950 17 17 
Question: 17 1223 17 1071 17 17 
Question: 1187 521 1071 612 17 675 1083 17 
Question: 475 285 480 427 376 419 316 17 
Question: 140 1223 953 1071 678 17 452 
Question: 1190 521 1071 107 748 1093 930 1071 367 17 
Question: 1195 288 1071 1053 17 646 17 17 
Question: 1194 140 702 1078 24 735 
Question: 1191 131 17 285 1178 849 682 
Question: 1189 1199 1071 1148 17 99 17 
Question: 521 17 17 17 
Question: 17 1223 352 475 1078 164 507 
Answer: 523 521 231 5 754 
Answer: 480 53 289 447 1069 403 81 
Answer: 1047 480 140 83 1223 1202 1070 1058 
Answer: 1071 650 134 1019 521 18 378 116 93 
Answer: 100 17 17 492 1071 17 
Answer: 1083 78 1176 1211 137 54 350 87 
Answer: 480 833 17 1071 543 17 17 
Answer: 1071 17 521 17 1093 1013 492 54 474 
Answer: 1219 1071 405 17 523 1199 820 1096 
Answer: 156 555 17 352 1070 683 638 
Answer: 17 521 1071 144 675 416 
Answer: 771 233 58 1205 682 982 17 1093 491 171 17 
Answer: 464 521 18 911 423 403 1224 838 
Answer: 1223 140 384 638 875 682 1071 678 287 974 
Answer: 100 1071 303 17 419 1186 1093 17 
Answer: 729 1176 268 137 17 17 
Answer: 1223 623 1013 1202 809 484 1223 73 655 1093 793 
Answer: 1071 1021 688 85 17 53 17 
Answer: 85 1071 17 1117 42 73 492 307 985 17 1202 151 
Answer: 156 480 1199 17 1071 814 638 17 
Answer: 480 106 1071 230 17 521 188 58 17 638 236 
Answer: 1223 17 889 1224 881 58 771 17 17 
Answer: 1219 47 1071 869 252 580 220 
Answer: 480 17 17 162 1071 17 17 17 
Answer: 756 492 17 48 264 612 593 
Answer: 18 795 592 140 802 424 17 
Answer: 590 1047 1223 460 17 1071 220 557 1161 
Answer: 1071 657 134 76 492 20 3 17 
Answer: 123 538 460 17 136 809 521 17 967 403 17 
Answer: 555 17 968 1070 1016 137 1016 
Answer: 988 913 523 521 12 136 606 521 17 
Answer: 480 344 1087 17 18 1182 
Answer: 1047 480 140 438 1088 1224 82 1202 1223 
Answer: 1223 140 713 492 1071 804 582 105 1071 132 
Answer: 1084 1012 73 17 100 1071 978 521 985 166 17 
Answer: 416 17 1071 1208 17 541 1091 75 
Answer: 403 1062 656 480 320 17 1093 1071 1158 
Answer: 1071 1030 240 521 657 17 
Answer: 1219 1070 70 521 1152 403 17 1071 1058 
Answer: 480 140 156 995 981 1093 462 1223 401 50 
Answer: 1071 17 534 521 231 1071 17 132 
Answer: 438 1093 1071 25 951 707 1093 871 1224 722 
Answer: 498 492 17 17 1223 888 58 358 164 
Answer: 1178 17 471 1071 194 492 1071 587 17 
Answer: 523 623 99 18 946 522 17 1071 236 419 17 
Answer: 1223 1199 652 71 419 1071 383 17 
Answer: 480 384 827 817 305 1093 1205 1202 403 17 17 
Answer: 480 749 1093 1056 18 17 657 637 
Answer: 1219 1070 74 17 17 991 
Answer: 464 73 1071 17 1017 1093 860 1071 130 
Answer: 17 1071 1000 17 521 14 644 682 1083 468 
Answer: 1146 1071 179 435 124 17 1093 226 18 655 124 
Answer: 156 480 140 797 523 403 17 58 17 
Answer: 480 17 585 550 1088 685 17 
Answer: 18 933 372 17 17 514 17 1146 
Answer: 1224 592 140 72 1071 551 864 
Answer: 1123 17 403 120 395 85 441 17 
Answer: 1071 855 958 99 291 492 20 1125 1183 
Answer: 17 17 165 17 17 462 1202 399 
Answer: 480 140 17 1071 402 403 108 17 
Answer: 480 17 57 1071 201 388 103 86 18 17 
Answer: 480 17 171 1202 966 1124 1075 17 17 17 
Answer: 675 17 555 17 1114 1070 668 
Answer: 752 1031 1224 349 861 492 1071 383 764 
Answer: 17 17 244 701 1091 17 371 295 
Answer: 1223 140 17 698 1093 1071 17 403 17 
Answer: 480 1081 17 18 521 638 308 1068 17 
Answer: 1178 140 611 43 657 418 484 1070 1207 403 1223 
Answer: 594 730 772 172 403 791 136 523 140 99 17 
Answer: 480 1199 312 1223 1071 776 17 17 
Answer: 1071 540 17 682 303 521 1071 117 17 
Answer: 1223 623 652 1093 33 54 495 1093 692 1071 1007 812 
Answer: 480 1199 161 1071 17 329 17 58 36 1071 17 
Answer: 1077 521 18 17 112 682 302 396 
Answer: 586 460 1125 17 17 1093 17 17 
Answer: 646 373 17 17 521 17 17 17 
Answer: 403 17 480 17 833 553 694 17 
Answer: 163 17 17 521 18 762 1091 1093 1169 17 
Answer: 1219 136 1223 958 197 1071 17 759 682 17 17 
Answer: 464 521 18 815 258 17 1071 192 
Answer: 480 399 682 17 17 492 17 17 
Answer: 687 1071 209 709 1093 159 1224 1053 951 
Answer: 17 1071 17 312 17 1093 1223 893 668 
Answer: 1079 473 1071 17 194 85 1071 211 154 17 
Answer: 1070 423 1176 248 492 1071 543 1161 
Answer: 1224 17 521 17 403 964 1071 207 
Answer: 17 913 1071 520 456 18 17 139 
Answer: 1071 17 388 17 492 4 
Answer: 1219 47 228 381 73 17 17 659 
Answer: 480 67 403 62 17 555 17 17 
Answer: 523 463 1093 654 58 17 17 1093 17 1224 147 
Answer: 17 521 18 762 1177 1093 17 804 17 17 
Answer: 1071 17 1104 221 521 20 13 
Answer: 811 191 875 73 92 85 589 868 17 
Answer: 375 73 17 137 242 403 929 17 
Answer: 219 1199 222 1117 17 403 1070 17 
Answer: 1178 1018 652 1093 1066 17 17 403 17 223 
Answer: 1178 749 1093 544 1071 795 492 17 657 1218 
Answer: 1219 17 46 535 521 17 332 
Answer: 162 1071 575 668 403 17 330 
Answer: 17 480 583 171 492 907 
Answer: 1146 1224 17 705 592 1093 502 1071 17 682 565 
Answer: 480 958 460 1071 861 17 137 1096 
Answer: 1071 632 381 623 99 492 1071 681 95 266 
Answer: 850 1205 140 17 17 403 594 730 
Answer: 18 17 255 140 613 1071 17 
Answer: 480 772 1071 234 1073 403 171 
Answer: 1178 44 1093 847 1071 657 1161 137 17 316 
Answer: 1070 521 18 17 130 58 665 17 
Answer: 1047 555 17 17 1223 1088 1071 195 1016 137 1016 
Answer: 1060 521 17 1192 1128 521 17 
Answer: 438 1093 1071 37 709 1093 33 18 655 1148 
Answer: 480 889 17 137 162 576 17 58 17 
Answer: 1071 678 287 521 682 1071 255 764 
Answer: 523 623 99 17 137 232 694 17 947 17 
Answer: 686 88 17 460 96 24 1093 1071 946 17 
Answer: 697 436 862 17 1071 640 17 17 
Answer: 1071 987 1141 1199 17 17 
Answer: 17 140 17 433 18 17 1202 1152 481 
Answer: 1071 587 17 17 75 732 58 1148 376 
Answer: 17 17 17 17 17 675 17 18 239 
Answer: 1146 18 727 783 694 54 685 17 
Answer: 480 1199 94 17 1071 237 103 17 160 
Answer: 1053 577 73 1022 1131 17 
Answer: 594 113 624 993 1093 331 17 17 
Answer: 17 17 17 58 17 17 388 17 338 
Answer: 480 17 833 18 17 17 17 1201 17 
Answer: 1178 17 1071 17 876 137 1071 316 675 1071 1182 
Answer: 1219 17 521 684 17 1206 403 17 
Answer: 480 140 247 1071 17 1145 492 18 17 1165 
Answer: 532 521 1071 540 17 492 697 17 1053 
Answer: 1223 140 17 58 934 17 1093 1225 1071 400 
Answer: 480 460 17 18 138 403 657 17 85 5 754 
Answer: 1223 623 320 1071 17 17 403 18 658 17 
Answer: 466 612 1145 17 17 300 17 
Answer: 1178 958 516 1071 17 250 1061 1093 17 
Answer: 17 18 17 546 623 491 732 
Answer: 1178 17 1093 1209 523 17 137 1071 240 657 637 
Answer: 17 668 623 912 1091 492 1071 578 905 
Answer: 1219 1071 705 1199 17 1093 1071 35 682 380 
Answer: 523 17 17 75 1125 17 1093 422 17 
Answer: 687 1071 654 951 1093 198 1093 1071 1173 
Answer: 480 1199 566 1071 1059 492 1071 17 17 
Answer: 697 17 218 1154 73 682 1071 17 17 707 
Answer: 1071 17 521 17 1093 800 940 497 
Answer: 1146 435 614 17 1093 17 18 614 17 17 
Answer: 1127 946 521 17 17 403 17 
Answer: 1178 17 1093 460 1071 382 17 137 657 17 
Answer: 660 1070 521 665 1071 220 1177 1093 138 1071 66 620 
Answer: 555 17 17 1071 786 492 967 537 
Answer: 1223 140 384 447 417 17 682 17 58 17 
Answer: 1219 1223 140 17 245 1224 25 492 951 
Answer: 17 1071 1061 202 566 668 
Answer: 697 655 677 521 572 682 1071 926 396 675 132 17 
Answer: 523 17 1071 287 521 18 1205 492 794 
Answer: 1071 17 587 1148 521 17 18 795 592 
Answer: 1178 1146 442 56 1093 1107 1148 621 
Answer: 1071 17 521 1152 403 684 1218 419 1071 806 238 
Answer: 523 17 682 17 1201 58 565 1202 17 17 
Answer: 697 929 759 17 17 17 58 10 
Answer: 523 463 1093 17 17 17 1202 18 17 70 
Answer: 1146 435 181 17 1093 852 1224 541 181 616 
Answer: 480 1199 687 18 1046 1090 403 1223 893 93 
Answer: 1071 923 140 99 17 419 1071 255 1181 
Answer: 578 131 17 17 99 17 1071 17 
Answer: 35 1071 222 17 1093 1071 17 592 694 469 592 
Answer: 640 17 464 1146 1174 17 694 506 483 
Answer: 1071 723 1199 17 99 17 657 418 
Answer: 523 521 17 17 464 85 1071 17 
Answer: 1047 17 1205 682 1071 1148 17 17 
Answer: 17 1071 468 521 17 1071 17 17 
Answer: 161 484 17 17 73 17 225 875 
Answer: 480 140 959 1223 475 1071 17 17 905 
Answer: 1071 1148 376 577 73 682 1071 510 56 235 
Answer: 1178 1107 621 1093 608 732 58 1145 
Answer: 697 790 704 140 1056 17 675 1070 374 
Answer: 949 1071 17 325 17 1093 1071 503 266 
Answer: 1178 17 18 17 17 657 17 
Answer: 1224 236 958 17 909 1178 1146 17 85 17 
Answer: 555 17 126 17 1071 17 492 18 966 17 
Answer: 18 17 771 521 1093 1021 17 492 18 457 409 
Answer: 1141 1224 792 492 1071 25 17 927 
Answer: 1219 480 140 528 1071 1165 194 668 
Answer: 1079 73 1022 492 1071 17 400 682 1071 17 296 
Answer: 523 17 988 1066 17 73 1018 17 17 223 
Answer: 18 17 1061 1199 588 1071 17 17 
Answer: 1178 231 1146 17 79 697 203 751 
Answer: 1178 749 1093 614 1071 17 124 1083 418 
Answer: 17 477 456 18 17 682 950 17 17 
Answer: 1071 17 17 73 492 1071 1148 17 252 
Answer: 523 1149 20 11 604 675 17 682 17 
Answer: 1178 140 427 1148 376 17 17 17 
Answer: 697 678 17 452 521 492 1071 1198 862 
Answer: 1071 17 73 18 17 1006 1093 17 1071 17 17 
Answer: 161 1224 17 58 722 484 17 521 17 
Answer: 686 1071 37 899 140 702 1084 735 
Answer: 600 521 697 17 131 17 403 525 17 
Answer: 1178 1199 17 1071 1148 17 17 1071 847 
Answer: 1219 1178 865 17 403 17 17 
Answer: 164 492 1071 910 635 17 507 1202 302 696 
token_cnt length: 12
vocab_size = 1226
Final layer weights loaded from final_layer_weight.bin.
Embedding matrix loaded from file: embedding_matrix.bin
PositionalEncoding Constructor
Attention weights for layer 0 loaded from files.
FeedForward weights for layer 0 loaded from file.
Attention weights for layer 1 loaded from files.
FeedForward weights for layer 1 loaded from file.
Attention weights for layer 2 loaded from files.
FeedForward weights for layer 2 loaded from file.
Attention weights for layer 3 loaded from files.
FeedForward weights for layer 3 loaded from file.
Attention weights for layer 4 loaded from files.
FeedForward weights for layer 4 loaded from file.
Attention weights for layer 5 loaded from files.
FeedForward weights for layer 5 loaded from file.
Transformer initialized with 6 layers.
Do you want to start mini prompt mode? (Y/N): y

Enter a string (or type 'exit' to quit): Would you like some tea
Truncated tokens (max length 12): 17 1223 17 988 17 0 0 0 0 0 0 0 
Category probabilities:
Question: 0.816051
Answer: 0.183949
Predicted category: Question

Enter a string (or type 'exit' to quit): Yes I would like some tea
Truncated tokens (max length 12): 1219 480 17 17 988 17 0 0 0 0 0 0 
Category probabilities:
Question: 0.29911
Answer: 0.70089
Predicted category: Answer

Enter a string (or type 'exit' to quit): What is the clock 
Truncated tokens (max length 12): 1187 521 1071 17 0 0 0 0 0 0 0 0 
Category probabilities:
Question: 0.838811
Answer: 0.161189
Predicted category: Question

Enter a string (or type 'exit' to quit): The clock is 5 PM
Truncated tokens (max length 12): 1071 17 521 17 754 0 0 0 0 0 0 0 
Category probabilities:
Question: 0.272759
Answer: 0.727241
Predicted category: Answer

Enter a string (or type 'exit' to quit): Could I help you with that
Truncated tokens (max length 12): 17 480 462 1223 1202 1070 0 0 0 0 0 0 
Category probabilities:
Question: 0.664448
Answer: 0.335552
Predicted category: Question

Enter a string (or type 'exit' to quit): Yes pleas help me with this
Truncated tokens (max length 12): 1219 17 462 17 1202 1083 0 0 0 0 0 0 
Category probabilities:
Question: 0.0142372
Answer: 0.985763
Predicted category: Answer

Enter a string (or type 'exit' to quit):  


```


