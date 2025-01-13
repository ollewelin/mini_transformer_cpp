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
./transformer_app 
========================================================================================================
Transformer Test in Mini Format (C/C++) - No Use of ML Libraries
The goal is to build and understand the Transformer algorithm from scratch using pure C++.
========================================================================================================

Loaded vocabulary of size: 1748 from vocab.txt
Loaded 616 examples total (Questions + Answers).
Do you want to load an existing model parameter with embedding matrix from a file? (Y/N, y/n): n
Tokenized Dataset:
Question: 1700 1580 765 767 987 
Question: 693 89 1744 416 1586 
Question: 186 1744 675 1 1717 1571 1544 
Question: 1703 765 1559 966 175 1492 
Question: 1708 765 1559 1428 155 
Question: 1707 1 1571 97 
Question: 1704 1 410 1744 1223 
Question: 1702 1713 1559 904 1480 
Question: 765 767 642 1585 1205 1588 
Question: 1 1744 514 1558 46 
Question: 1700 765 1559 193 999 605 
Question: 693 186 700 716 959 248 1 
Question: 186 1744 1402 1 63 498 999 1558 617 
Question: 1703 186 700 558 947 1282 
Question: 1708 415 1559 1 1309 717 1559 1 
Question: 700 100 1700 1 1744 1 
Question: 1704 1169 798 1401 700 815 564 
Question: 1702 415 1559 1494 1014 
Question: 765 767 1330 1585 1615 20 987 
Question: 1 1744 1115 1259 1559 1198 
Question: 1700 765 1745 1020 1007 1559 327 1 
Question: 693 410 700 1147 586 17 774 753 
Question: 186 1744 281 1559 367 999 1559 1272 
Question: 1703 765 1559 141 1108 1585 1 92 677 
Question: 1708 410 1691 968 1585 1648 1122 717 1 
Question: 1707 186 675 1 1717 1571 1171 
Question: 1704 824 410 700 968 1585 737 
Question: 1702 765 1559 975 175 95 
Question: 765 1189 142 1556 772 
Question: 1 1744 1415 1558 482 
Question: 1700 765 1559 897 999 827 
Question: 693 1 410 1744 504 
Question: 186 1744 104 1 1717 959 1 
Question: 1703 1401 700 1055 959 194 
Question: 1708 89 1565 1438 879 1479 717 1559 1428 
Question: 1707 1 1559 1725 1 802 1739 
Question: 1704 1 410 1744 1223 586 1548 974 
Question: 1702 765 1559 346 586 1559 1507 
Question: 765 1571 1559 309 1689 1585 1 1559 1544 
Question: 1 1744 1115 1455 947 1 
Question: 1700 765 1559 1 173 717 1559 1725 
Question: 693 410 700 1275 959 1066 
Question: 186 1744 514 1559 275 999 729 717 1 
Question: 1703 1401 1691 687 1559 278 
Question: 987 959 1198 765 1571 1 1621 
Question: 1707 186 1 1571 1186 
Question: 1704 603 410 1744 558 1 1585 1648 
Question: 1702 89 1744 1112 1585 1542 17 1 
Question: 765 1558 17 1655 90 
Question: 1 1744 1 1559 1489 1585 1 1559 171 
Question: 1700 765 1559 893 1464 1 1007 1571 1313 
Question: 693 410 700 317 17 973 637 164 
Question: 186 1744 1173 1571 413 586 1 
Question: 1703 377 1744 815 19 865 816 
Question: 1708 765 959 246 1 17 1367 540 
Question: 1707 186 86 959 817 1267 
Question: 1704 1694 410 1744 1648 586 573 159 
Question: 1702 1713 1559 1256 128 1 
Question: 765 234 1 1 586 1 
Question: 1 1744 216 1559 583 1424 1115 
Question: 1700 765 1559 141 84 1585 1 1571 1161 
Question: 693 377 1744 634 1 717 1169 
Question: 186 1744 1611 1571 1377 586 1 
Question: 1703 410 700 1508 959 510 1264 
Question: 69 1708 415 959 1093 127 425 1438 1200 
Question: 1707 765 1559 871 288 1089 586 1528 
Question: 1704 1 410 1744 1569 765 947 446 
Question: 1702 1401 1691 1342 1559 975 904 
Question: 765 249 1025 1 142 586 1166 
Question: 1 1744 1391 1559 1150 1 
Question: 1700 765 1559 801 1 1007 437 
Question: 693 410 700 1022 1571 1473 1196 
Question: 1 186 1744 675 1 1 1566 268 486 
Question: 1703 186 700 1227 1566 1 
Question: 1708 410 869 672 1 
Question: 1707 765 1745 541 1 1 
Question: 1 1 1704 165 999 1 1726 1744 1223 
Question: 959 1 1702 765 1559 141 1580 1585 1674 1 
Question: 765 767 1087 1585 167 1 1 
Question: 1 1744 709 1559 275 1717 63 498 
Question: 1700 765 1745 1497 586 1 
Question: 693 410 700 216 1559 1537 1388 
Question: 186 1744 596 1 1559 455 1 
Question: 1703 410 1567 691 1559 1 278 
Question: 1708 765 1558 917 362 
Question: 1707 765 1 586 1410 1559 294 
Question: 1704 1093 940 667 1559 141 185 
Question: 1702 377 1559 1 564 1014 
Question: 765 54 1559 339 1 1 1 
Question: 1 1744 1 1745 802 1 
Question: 1700 765 1559 141 1068 1585 1 959 198 
Question: 693 415 1010 133 17 142 1183 1456 
Question: 186 1744 1664 1559 1598 311 
Question: 1703 186 700 815 947 19 1193 274 
Question: 1708 89 1 543 1 179 353 
Question: 410 1744 1 1707 765 642 1585 313 1559 1615 1 
Question: 1704 1552 1 672 1691 984 1 1742 
Question: 1702 765 1559 1171 509 1585 807 
Question: 765 794 999 1 489 586 1571 1316 
Question: 1 1744 221 1559 852 586 486 
Question: 1700 765 1745 541 1169 798 
Question: 693 410 700 737 1571 1 1007 835 
Question: 186 1744 1 1559 1264 179 1588 
Question: 1703 377 1744 1363 1559 936 553 802 
Question: 1708 410 1446 1081 1145 1249 1721 
Question: 1707 186 907 973 1547 1 
Question: 1704 255 1 410 1744 1 141 
Question: 1702 1713 1744 1245 1559 975 1665 
Question: 765 1571 17 171 1025 1 1 
Question: 1 1744 661 1 1577 1559 280 
Question: 1700 765 1559 1 143 1546 69 1627 
Question: 410 1567 1 693 410 700 35 17 973 1650 1585 1559 1537 
Question: 186 1744 363 1745 1723 586 246 1 
Question: 1703 410 700 558 1559 1002 414 
Question: 1708 765 1559 1043 1 717 1610 
Question: 1707 765 112 1585 23 1559 1383 1 
Question: 1704 1265 1 1559 806 1450 246 
Question: 186 1 1642 1559 1 
Question: 765 1565 17 1 117 586 1 
Question: 1 1744 1522 1559 871 1 
Question: 1700 765 1559 1 334 1 1 
Question: 693 410 700 301 1571 552 1585 1075 
Question: 186 1744 317 17 121 999 1559 341 
Question: 1703 186 700 1363 1559 1537 852 
Question: 1708 410 1446 1459 923 853 1 
Question: 1707 1688 1559 564 1089 1585 1 1 496 
Question: 1704 1 1443 410 1744 1145 
Question: 1702 410 1691 1 1585 1 120 611 1559 239 
Question: 765 1565 17 1 1723 586 1 1 
Question: 1 1744 359 693 1585 1648 1571 1591 
Question: 1700 765 1559 801 1 717 1031 1 1537 
Question: 693 410 700 272 17 581 754 17 1746 552 
Question: 186 1744 1342 17 1 184 586 975 1695 
Question: 1703 765 1559 141 1472 1585 1 1559 1 
Question: 1708 415 905 1647 1467 432 1481 
Question: 1707 1401 1691 760 1585 1559 1 1385 
Question: 1704 84 1 1585 142 1084 
Question: 1702 410 1744 1569 1691 186 1727 1 1571 1171 
Question: 765 767 1 1 1559 1 1 
Question: 1 1744 281 1559 1 37 
Question: 1700 765 1559 1 219 1580 586 1571 371 
Question: 693 410 700 282 1585 1559 1679 
Question: 186 1744 1035 1559 1545 586 1571 1 
Question: 1703 186 700 1214 19 1745 1 1657 
Question: 1708 410 1691 968 1585 1406 17 1 1 
Question: 1707 186 1278 17 1 717 637 
Question: 1704 571 999 835 410 1744 1223 586 1 
Question: 1702 186 1691 1 1559 556 365 1 
Question: 765 1558 1559 309 1647 999 1559 80 
Question: 1 1744 1 1559 1161 717 1414 1 
Question: 1700 765 1559 141 606 1281 1585 815 339 1 
Question: 693 186 700 356 959 24 1 
Question: 186 1744 596 1 1559 1547 288 836 
Question: 1703 765 1559 973 1001 844 
Question: 1708 765 1559 414 720 
Question: 1707 765 1559 871 1650 999 1571 542 
Question: 1704 66 1591 410 1691 1648 
Question: 1702 415 1559 1 1 
Question: 765 1745 1537 267 1717 1 
Question: 1 1744 1 1 999 1031 1362 1125 
Question: 1700 765 1559 141 1689 1585 84 1 
Question: 693 410 700 1251 17 637 262 
Question: 186 1744 1014 17 1528 1579 586 1 
Question: 1703 186 700 423 1559 1 1355 
Question: 1708 765 1559 1162 1 1438 853 
Question: 1707 1401 700 37 717 1559 313 1 
Question: 1704 703 410 949 1 677 1648 
Question: 1702 410 1744 1110 1007 1 1559 1067 
Question: 765 767 1 107 1745 845 987 
Question: 1 1744 675 1 365 1559 1650 1 
Question: 1700 765 1559 141 1 1585 634 1585 1559 1 
Question: 693 410 700 1230 316 1647 
Question: 186 1744 359 693 1585 1326 1566 1 
Question: 1703 186 700 1363 1559 1650 546 852 
Question: 1708 89 1691 1604 1566 919 
Question: 1707 186 1542 1 999 1559 973 542 
Question: 1704 477 1 410 700 968 1585 1386 
Question: 1702 765 1559 975 246 1 1 
Question: 765 959 339 1330 1717 1571 280 
Question: 1 1744 166 1 1559 311 1 
Question: 1700 765 1559 141 1140 586 1066 1493 
Question: 693 410 700 1642 959 1167 727 
Question: 186 1744 775 1559 1669 278 717 567 1 
Question: 1703 410 1691 1494 1 1 
Question: 1708 765 1559 246 314 1490 863 
Question: 1707 765 642 1585 872 1559 1 1 
Question: 1704 289 1113 89 1691 1652 
Question: 1702 89 1691 1 1559 1 164 
Question: 765 1565 17 1 586 1387 1 1 
Question: 1 1744 1 1559 1 1 
Question: 1700 765 1559 905 1 999 1571 1 
Question: 693 410 700 626 546 611 462 1 
Question: 186 1744 1391 1559 1002 1 661 
Question: 1703 765 1559 141 1108 1585 1363 1559 533 1 
Question: 1708 415 1559 1537 1 959 1 1 
Question: 1707 186 1038 1566 23 1088 
Question: 1704 172 1591 410 1691 1247 1007 
Question: 1702 1713 1559 1650 1 128 1 
Question: 765 1 1 1 
Question: 1 1744 514 693 1566 233 744 
Question: 1700 765 1745 541 157 69 1708 
Question: 693 410 700 1620 970 1 1 
Question: 186 1744 1223 17 643 1 586 816 686 
Question: 1703 765 1559 966 1 1 1 1 
Question: 1708 410 1691 1 107 1 
Question: 1707 1688 1559 564 1 1585 1 1559 1564 999 1 
Question: 1704 1592 89 141 586 1580 1 
Question: 1702 765 1559 141 1580 1585 1 17 1 
Question: 765 767 1135 1585 1615 539 1556 1559 1464 999 828 
Question: 1 1744 1391 1745 1 1007 1 216 
Question: 1700 765 1559 1 143 1 69 1 
Question: 693 410 700 1177 959 1091 727 1011 
Question: 1703 186 700 558 1623 586 1 1 
Question: 1708 415 1 508 1702 767 1 
Question: 1707 765 1 1559 1 999 1 274 
Question: 1704 938 82 410 1744 1648 586 1 
Question: 1702 1713 1 797 1007 1 
Question: 765 1565 827 1 437 
Question: 1 1744 514 1559 1 999 153 1 
Question: 1700 765 1559 949 1 1689 1585 815 17 973 798 
Question: 693 410 700 1 959 688 1711 970 1 
Question: 1703 186 700 1 1 1164 1 
Question: 1708 765 1 1 586 1 
Question: 1707 765 1559 949 1 1 999 1559 1 1 
Question: 1704 1 1 1 1559 949 1580 
Question: 1702 410 149 1 923 
Question: 765 1 1 586 1 1 
Question: 693 410 700 1 263 486 717 1189 
Question: 186 1744 1 643 160 1007 1 
Question: 1708 415 1559 1 1 1 965 1559 690 
Question: 1703 186 700 558 1 1443 586 1094 1 
Question: 1700 89 1559 1 999 816 17 1 1 
Question: 1707 1 1559 564 1 1 
Question: 1704 1 295 1585 142 1 1 
Question: 1702 765 1559 141 1580 1585 1 1559 1 1 
Question: 765 1565 17 1689 1585 1230 464 1 107 688 
Question: 1 1744 1522 1559 786 1 999 1 1564 999 1 
Question: 693 410 700 1 69 872 959 1 185 
Question: 1703 765 1559 141 1108 1585 1 965 1 
Question: 1708 410 1 672 378 1 1 
Question: 1707 765 1 586 1387 1 751 1 
Question: 1704 243 1493 1384 410 1744 1223 
Question: 1702 415 1559 975 1 1 1 
Question: 765 1565 17 1 661 586 1652 835 
Question: 186 1744 1180 1583 586 1729 17 1 1291 
Question: 1700 765 1559 1 1689 1585 1 1 747 
Question: 693 410 700 1480 17 1 611 1 
Question: 1703 186 700 558 1 606 1 1011 
Question: 1708 410 1446 1 1 432 1 
Question: 1707 1 1559 1 1 
Question: 1704 1 1 765 949 1 586 1 
Question: 1702 765 1559 141 1580 1585 1674 1559 1 1 
Question: 765 1565 63 439 1689 1585 1 1 1 
Question: 1 1744 514 693 1 89 1 
Question: 1700 765 1559 1 143 17 1 69 63 1 
Question: 693 410 700 1644 1559 1 717 959 1 
Question: 1703 765 1559 141 1472 586 1 717 1571 1 
Question: 1708 410 1446 1081 672 17 1 999 1 
Question: 1707 1 1559 564 1 717 686 
Question: 1704 1166 1 89 1 1 
Question: 1702 1713 1193 1 133 1 
Question: 765 767 1135 1585 1227 452 1 1 
Question: 1 1744 1684 1 1577 1387 1 17 688 1 1537 
Question: 1700 765 1559 1 999 1559 751 999 1 1 
Question: 693 410 700 1 17 1 1 999 1 
Question: 1703 186 700 558 1 1 1 
Question: 1708 415 1 1 1007 1 
Question: 1707 89 1559 1593 1 717 1 464 1586 
Question: 1704 1625 999 1 765 141 586 1 
Question: 1702 1401 700 1260 959 194 1 
Question: 765 1 142 1556 1030 1 999 504 
Question: 1 1744 514 1559 1 143 1 69 1 464 1451 
Question: 1700 765 1559 141 1689 1585 1480 816 19 1 
Question: 693 410 700 1620 959 1 1702 767 1 1 1007 
Question: 1703 186 700 558 1246 974 1451 
Question: 1708 410 1691 1 1 1007 437 
Question: 1707 1 1559 1564 999 629 1 
Question: 1704 650 365 1443 765 141 586 1 
Question: 1702 377 1 564 797 1007 1559 1 
Question: 765 767 1 816 1585 246 717 1 
Question: 1 1744 1223 17 1497 586 1 1580 1 
Question: 1700 765 1559 141 1689 1585 1494 380 1 1 
Question: 693 410 700 1278 1 1 1717 1443 
Question: 1703 186 700 558 842 1 1 
Question: 1708 415 959 1093 1 1438 1200 
Question: 1707 89 1559 786 1 717 1559 686 999 1 
Question: 1704 160 89 489 586 1 1 
Question: 1702 765 1559 141 1580 1585 1 1 1 
Question: 765 1565 17 1689 1585 1142 1 1 
Question: 1 1744 363 1559 1489 1585 317 17 176 1110 
Question: 1700 765 1559 1 999 1 1 1459 
Question: 693 410 700 716 959 1183 1 1 
Question: 1703 186 700 1 17 1 717 1571 232 
Question: 1708 410 1479 1 717 1559 1 1428 
Question: 1707 1698 1559 1 999 1559 828 1 
Question: 1704 1443 765 141 586 1669 1 
Question: 1702 1713 1559 975 873 1 1 1 
Question: 765 1565 17 1 1585 1 1 
Question: 1 1744 1180 17 1 586 1 754 17 973 1 
Question: 1700 765 1559 949 1 1689 1585 426 
Question: 693 410 700 1 1721 69 1091 827 1 
Question: 1703 765 1559 966 824 
Question: 1708 410 1446 1 426 1007 1559 818 1405 999 1559 1313 
Question: 1707 765 1559 327 1 999 1559 1 1 
Question: 1704 169 1 716 1166 
Question: 1702 377 1559 1 1 136 
Question: 765 1565 17 643 1689 1585 1152 1 
Question: 1 1744 1 1690 1585 1 17 1433 1 
Answer: 767 765 328 6 1119 
Answer: 700 60 416 652 1557 586 101 
Answer: 1529 700 186 104 1744 1717 1558 1544 
Answer: 1559 966 175 1492 765 17 550 154 118 
Answer: 132 1 1 717 1559 1 
Answer: 1571 97 1688 1730 179 63 511 111 
Answer: 700 1223 1 1559 806 1 1 
Answer: 1559 904 765 1 1585 1480 717 63 692 
Answer: 1740 1559 588 1 767 1713 1205 1588 
Answer: 213 822 1 514 1558 1009 947 
Answer: 1 765 1559 193 999 605 
Answer: 1140 334 69 1721 1007 1433 1 1585 716 248 1 
Answer: 677 765 17 1332 617 586 1745 1233 
Answer: 1744 186 558 947 1282 1007 1559 1002 414 1422 
Answer: 132 1559 437 1 611 1699 1585 1 
Answer: 1080 1688 388 179 1 1 
Answer: 1744 922 1480 1717 1189 707 1744 89 973 1585 1169 
Answer: 1559 1494 1016 107 13 60 1 
Answer: 107 1559 1 1615 45 89 717 444 1438 1 1717 207 
Answer: 213 700 1713 1 1559 1198 947 1 
Answer: 700 138 1559 327 1 765 270 69 1 947 339 
Answer: 1744 1 1302 1745 1291 69 1140 1 1 
Answer: 1740 54 1559 1272 367 855 309 
Answer: 700 1726 1 222 1559 1 1 1 
Answer: 1122 717 1 55 383 905 878 
Answer: 17 1171 876 186 1180 619 1 
Answer: 874 1529 1744 672 1 1559 309 824 1665 
Answer: 1559 975 175 94 717 19 1 1 
Answer: 163 799 672 1 177 1189 765 1 1414 586 1 
Answer: 822 1 1415 1558 1488 179 1488 
Answer: 1446 1335 767 765 1 177 897 765 1 
Answer: 700 504 1576 1 17 1695 
Answer: 1529 700 186 640 1577 1745 103 1717 1744 
Answer: 1744 186 1055 717 1559 1183 861 137 1559 173 
Answer: 1572 1479 89 1673 132 1559 1428 765 1438 237 1 
Answer: 605 1 1559 1725 1 802 1580 92 
Answer: 586 1548 974 700 468 837 1585 1559 1662 
Answer: 1559 1507 346 765 975 1 
Answer: 1740 1558 84 765 1655 586 1 1559 1544 
Answer: 700 186 213 1455 1432 1585 675 1744 582 57 
Answer: 1559 1 790 765 328 1559 1 173 
Answer: 640 1585 1559 24 1388 1046 1585 1275 1745 1066 
Answer: 729 717 1 1 1744 1299 69 520 233 
Answer: 1691 1 687 1559 278 717 1559 871 1 
Answer: 767 922 128 17 1383 766 1 1559 339 611 1 
Answer: 1744 1713 968 85 611 1559 557 1 
Answer: 700 558 1213 1201 439 1585 1721 1717 586 613 1 
Answer: 700 1110 1585 1542 17 1 975 946 
Answer: 1740 1558 90 1 1 1449 
Answer: 677 89 1559 1 1489 1585 1261 1559 171 
Answer: 1 1559 1464 1 765 1 953 1007 1571 683 
Answer: 1648 1559 260 637 164 1 1585 317 17 973 164 
Answer: 213 700 186 1173 767 586 1 69 1 
Answer: 700 1 865 816 1577 1011 1 
Answer: 17 1367 540 1 1 756 1121 1648 
Answer: 1745 876 186 86 1559 817 1267 
Answer: 1622 1 586 158 574 107 643 1 
Answer: 1559 1256 1401 128 419 717 19 1624 1696 
Answer: 1 1 234 1 1 675 1717 579 
Answer: 700 186 469 1559 583 586 142 1 
Answer: 700 1726 67 1559 287 564 135 109 17 1 
Answer: 700 1 248 1717 1413 1623 1563 1 1 1 
Answer: 999 1 822 1 1611 1558 987 
Answer: 1115 1508 1745 510 1264 717 1559 557 1132 
Answer: 1 1 355 1037 1580 1 539 425 
Answer: 1744 186 1 1032 1585 1559 1 586 1 
Answer: 700 1569 1 17 765 947 446 1556 1 
Answer: 1691 186 903 46 975 609 707 1558 1724 586 1744 
Answer: 879 1081 1145 249 586 1166 177 767 186 128 1 
Answer: 700 1713 455 1744 1559 1150 1 1 
Answer: 1559 801 1 1007 437 765 1559 155 1 
Answer: 1744 922 968 1585 35 63 722 1585 1022 1559 1473 1196 
Answer: 700 1713 221 1559 1 484 1 69 38 1559 246 
Answer: 1565 765 17 1 147 1007 436 575 
Answer: 869 672 1624 1 1 1585 1 52 
Answer: 959 541 1 1 765 1 1 1 
Answer: 586 1 700 1 1223 819 1025 1 
Answer: 224 1 1 765 17 1129 1580 1585 1674 1 
Answer: 1740 177 1744 1401 281 1559 1 1125 1007 1 1 
Answer: 677 765 17 1199 373 1 1559 275 
Answer: 700 579 1007 1 1 717 1 1 
Answer: 1014 1559 296 1049 1585 216 1745 1537 1388 
Answer: 1 1559 1 455 1 1585 1744 1307 987 
Answer: 1567 691 1559 1 278 107 1559 298 210 1 
Answer: 1558 617 1688 362 717 1559 806 1665 
Answer: 1745 1 765 1 586 1410 1559 294 
Answer: 1 1335 1559 764 667 17 1 185 
Answer: 1559 1 564 1 717 1 
Answer: 1740 54 320 553 89 1 1 977 
Answer: 700 81 586 75 1 822 1 1 
Answer: 767 676 1585 970 69 1 1 1585 1 1745 198 
Answer: 1 765 17 1129 1689 1585 1 1183 1 1 
Answer: 1559 1 1598 311 765 19 1 
Answer: 1193 274 1282 89 117 107 873 1271 1 
Answer: 543 89 1 179 353 586 1362 1 
Answer: 308 1713 313 1615 1 586 1558 1 
Answer: 1691 1490 968 1585 1552 1 1 586 1 314 
Answer: 1691 1110 1585 807 1559 1171 717 1 975 1739 
Answer: 1740 1 50 794 765 1 489 
Answer: 222 1559 849 987 586 1 486 
Answer: 1 700 862 248 717 1328 
Answer: 1648 1745 1 1043 876 1585 737 1559 1 1007 835 
Answer: 700 1401 672 1559 1264 1 179 1588 
Answer: 1559 936 553 922 128 717 1559 1006 121 385 
Answer: 1249 1721 186 1 445 586 879 1081 
Answer: 17 1 370 186 907 1559 1 
Answer: 700 1145 1559 337 1561 586 248 
Answer: 1691 48 1585 1245 1559 975 1665 179 1 462 
Answer: 1558 765 17 1 171 69 984 1 
Answer: 1529 822 1 1684 1744 1577 1559 279 1488 179 1488 
Answer: 1546 765 1 1705 1627 765 1 
Answer: 640 1585 1559 39 1049 1585 35 17 973 1650 
Answer: 700 1302 246 179 222 850 1 69 1 
Answer: 1559 1002 414 765 1007 1559 370 1132 
Answer: 767 922 128 1 179 332 1025 1 1384 1 
Answer: 1012 112 1 672 122 23 1585 1559 1383 1 
Answer: 1031 638 1265 1 1559 949 1 246 
Answer: 1559 1443 1642 1713 1 1 
Answer: 1 186 1 634 17 1 1717 1655 702 
Answer: 1559 871 1 1 92 1084 69 1650 546 
Answer: 1 1 1 1 1 999 1 17 344 
Answer: 1648 17 1075 1157 1025 63 1011 1 
Answer: 700 1713 120 1 1559 341 135 1 217 
Answer: 1537 852 89 1495 1630 1 
Answer: 879 149 923 1452 1585 488 250 1 
Answer: 1 1 1 69 1 1 564 1 496 
Answer: 700 1 1223 17 1 1 1 1715 1 
Answer: 1691 1 1559 1 1283 179 1559 462 999 1559 1695 
Answer: 1740 1 765 1010 1 1723 586 1 
Answer: 700 186 359 1559 1591 1647 717 17 1 1669 
Answer: 783 765 1559 801 1 717 1031 1 1537 
Answer: 1744 186 1 69 1369 1 1585 1746 1559 581 
Answer: 700 672 1 17 184 586 975 1 107 6 1119 
Answer: 1744 922 468 1559 1 1 586 17 976 1 
Answer: 681 905 1647 1 1 432 1 
Answer: 1691 1401 760 1559 1 365 1547 1585 1 
Answer: 1 17 1 809 922 716 1084 
Answer: 1691 1 1585 1727 767 1 179 1559 346 975 946 
Answer: 1 987 922 1334 1580 717 1559 853 1326 
Answer: 1740 1559 1043 1713 1395 1585 1559 37 1007 552 
Answer: 767 1 1 92 1624 1 1585 616 219 
Answer: 1014 1559 970 1388 1585 282 1585 1559 1679 
Answer: 700 1713 836 1559 1545 717 1559 1 1 
Answer: 1031 1 306 1657 89 1007 1559 1 1 1046 
Answer: 1559 1 765 1 1585 1177 1376 727 
Answer: 1648 637 909 1 1585 1278 17 909 1 1 
Answer: 1626 1383 765 1 1 586 1 
Answer: 1691 1 1585 672 1559 556 1 179 975 1 
Answer: 978 1558 765 984 1559 309 1689 1585 184 1559 80 917 
Answer: 822 1 1 1559 1161 717 1414 798 
Answer: 1744 186 558 652 606 1 1007 1 69 1 
Answer: 1740 1744 186 1 356 1745 24 717 1388 
Answer: 1 1559 1547 288 836 987 
Answer: 1031 973 1001 765 844 1007 1559 1359 575 999 173 1 
Answer: 767 1 1559 414 765 17 1721 717 1170 
Answer: 1559 1 871 1650 765 1 17 1171 876 
Answer: 1691 1648 644 66 1585 1602 1650 919 
Answer: 1559 1 765 1655 586 1010 1739 611 1559 1186 343 
Answer: 767 1 1007 1 1715 69 835 1717 1 1 
Answer: 1031 1362 1125 1 1 1 69 1 
Answer: 767 676 1585 1 1 1 1717 17 1 84 
Answer: 1648 637 262 1 1585 1251 1745 802 262 911 
Answer: 700 1713 1014 17 1528 1579 586 1744 1307 118 
Answer: 1559 1355 186 128 1 611 1559 370 1694 
Answer: 853 172 1 1 128 1 1559 1 
Answer: 37 1559 313 1 1585 1559 1 876 1025 685 876 
Answer: 949 1 677 1648 1680 246 1025 743 704 
Answer: 1559 1067 1713 1 128 1 975 609 
Answer: 767 765 1 1 677 107 1559 1 
Answer: 1529 1 1721 1007 1559 1650 1 1 
Answer: 1 1559 683 765 1 1559 1 1 
Answer: 221 707 1 1 89 1 316 1282 
Answer: 700 186 1402 1744 693 1559 1 1 1326 
Answer: 1559 1650 546 852 89 1007 1559 750 66 338 
Answer: 1691 1602 919 1585 900 1084 69 1647 
Answer: 1031 1165 1041 186 1542 1 999 1558 542 
Answer: 1386 1559 1 477 1 1585 1559 738 385 
Answer: 1691 1 17 246 1 975 1 
Answer: 1745 339 1401 1 1330 1691 1648 461 107 1285 
Answer: 822 1 166 1 1559 1 717 17 1413 1 
Answer: 17 1 1140 765 1585 1494 1 717 17 668 593 
Answer: 1642 1745 1167 717 1559 24 1 1360 
Answer: 1740 700 186 775 1559 1669 278 987 
Answer: 1567 89 1495 717 1559 1 581 1007 1559 1392 426 
Answer: 767 1 1446 1552 1 89 1490 1 1 314 
Answer: 17 1 1547 1713 872 1559 1 246 
Answer: 1691 328 1648 1 99 1031 289 1113 
Answer: 1691 1110 1585 909 1559 1 164 1571 609 
Answer: 1 695 667 17 1 1007 1387 1 1 
Answer: 1559 1 1 89 717 1559 1650 1 367 
Answer: 767 1651 19 1 895 999 1 1007 1 
Answer: 1691 186 626 1650 546 1 1 1 
Answer: 1031 1002 1 661 765 717 1559 1712 1265 
Answer: 1559 1 89 17 1 1472 1585 1 1559 1 1 
Answer: 221 1745 1 69 1066 707 1 765 1 
Answer: 1012 1559 39 1316 186 1038 1572 1088 
Answer: 891 765 1031 1 172 1591 586 772 1 
Answer: 1691 1713 1184 1559 1650 1 1 1559 1245 
Answer: 1740 1691 1268 1 586 1 1 
Answer: 233 717 1559 1331 942 1 744 1717 436 1030 
Answer: 1559 141 1689 1585 1 1571 765 1585 166 767 754 1 1 1545 
Answer: 1571 1 1 667 606 1056 1 1 10 1119 
Answer: 1559 1 765 509 1585 429 805 1 
Answer: 17 651 196 1644 186 675 716 1084 586 1 
Answer: 700 1 1652 1 1585 1 1745 1545 947 1 
Answer: 1 558 17 1 661 717 1559 1165 1 
Answer: 1744 186 634 1566 553 179 1 1559 1392 426 
Answer: 1740 700 186 184 1744 120 717 3 1 
Answer: 1031 1607 1 1 1 1624 1696 1585 269 
Answer: 1559 1 1 63 475 1 999 1 1082 1089 
Answer: 1744 186 288 1559 1 107 1 586 1 
Answer: 1559 973 1171 1 667 134 1 1585 1 
Answer: 221 1745 1 581 586 1559 936 455 
Answer: 1744 186 1 1559 1 1007 1 
Answer: 1559 1 1458 1586 765 1 1 
Answer: 874 1529 1585 784 1 999 1745 1 1 
Answer: 17 1 186 675 1602 1745 1 1 947 1 
Answer: 1 968 1585 1149 1655 702 107 1559 1 586 1663 
Answer: 1559 1 806 1665 667 1 127 1 
Answer: 1571 1 1 1268 17 947 1 1383 1585 664 1559 840 
Answer: 700 1223 1 1 586 17 1 1 1007 698 686 
Answer: 1559 1 1 17 1 606 1 586 973 1 
Answer: 1031 1547 765 328 1 1571 704 619 
Answer: 1559 966 1 765 1 29 1559 1 
Answer: 1744 922 968 1585 1235 1559 1046 586 1559 1643 1585 1 
Answer: 1691 1401 1 1559 1 179 1559 462 999 1559 1695 
Answer: 1 17 652 97 19 1571 1594 1007 1 
Answer: 1622 1652 17 1066 876 586 142 1362 
Answer: 1559 141 1108 1585 1670 1559 1 765 1559 1 1055 
Answer: 17 1443 1642 922 1278 1559 1 766 
Answer: 1559 1606 1 94 107 1113 7 107 1571 1 
Answer: 1744 186 1508 1745 1 1577 1559 1011 1132 
Answer: 1571 542 1713 128 1 717 1559 975 1443 1642 
Answer: 1585 1230 1 1 1652 579 1 1 1 1025 1 
Answer: 1691 1 546 1585 716 1031 1 1 
Answer: 1744 186 1 1745 1615 196 107 1559 1 1 
Answer: 586 1 700 1223 1 1032 1571 1 661 
Answer: 1031 824 667 17 1 1360 586 1 160 
Answer: 1559 484 1 1585 128 1 179 936 1 
Answer: 1744 186 1 17 184 611 1031 1547 1 1 1 
Answer: 1559 494 1713 128 1 1007 1031 1 1 
Answer: 1 221 1559 1 1156 135 1410 17 294 
Answer: 17 1 1047 999 1 765 489 586 1 1 
Answer: 1571 1 765 439 1585 582 69 1 586 1 
Answer: 1559 184 1713 802 1 1 1 
Answer: 784 63 1 1007 1559 1428 1 586 17 1 1585 1472 1 1479 
Answer: 1115 1 1745 1 586 75 1 
Answer: 1571 940 765 1 586 1 1 69 1 1084 
Answer: 1691 968 1585 626 54 1559 546 135 1 1559 365 
Answer: 1571 1 765 1 177 1 17 1 854 1556 1559 683 
Answer: 1744 186 1602 1745 1 1577 1559 834 717 1745 1 455 
Answer: 1691 1223 1 1745 1 1443 1 
Answer: 1571 1594 765 1 1 717 1559 1 1 
Answer: 1559 1 1 1585 634 1 1 3 60 1438 640 1 
Answer: 1571 1 1401 716 1745 1357 1290 1 
Answer: 1 1 142 1585 1173 1745 1721 1 135 1507 
Answer: 1559 973 1 667 1 652 1 586 1 1 
Answer: 1571 603 765 1 69 1724 1697 586 1433 1 
Answer: 1744 186 775 1031 1011 1 586 1 1 1007 1571 1594 
Answer: 874 1529 1585 1 1 707 1744 1110 1585 1466 1580 1 1586 
Answer: 1571 542 765 1490 717 1 1553 1438 1 1446 1 
Answer: 1744 186 1334 947 179 1 63 1 1511 1110 
Answer: 1571 255 1 1724 1697 586 1 1 
Answer: 1558 1606 765 1 179 3 1 1 1585 1 1721 
Answer: 1744 186 558 1 1164 107 1559 1 884 1007 1 1 
Answer: 874 1529 1585 1 1559 1 1585 1 135 1 1559 1 
Answer: 1559 966 1 1 765 1624 927 1 1007 1571 1313 
Answer: 1740 1559 1 586 1559 1 89 1490 117 1011 
Answer: 1559 573 1688 1 1 1585 1 1692 1 
Answer: 1744 186 1648 1 1 69 1 1585 1 1559 1 
Answer: 1559 842 1055 765 17 652 1472 586 17 1 1007 1 345 
Answer: 1559 1 89 1 717 1559 1 1 582 1559 1489 1 
Answer: 1 968 17 1 1585 1648 1571 371 717 73 1 
Answer: 1605 765 1 1556 1 1 1585 63 1 1007 1559 683 
Answer: 1559 966 108 765 844 1 1 1559 1 1 
Answer: 1571 940 999 1559 1 1 1 1717 17 1 1 
Answer: 1559 1 1 1 92 1 717 1571 1236 
Answer: 1 968 1745 1 69 17 1 1585 1615 1585 1558 1 
Answer: 1559 1 1 107 3 1119 177 1 1559 802 1 107 1 1119 
Answer: 1559 175 1 765 1 69 1 968 1 216 
Answer: 1559 1 1 766 922 128 1 1585 17 1 1 
Answer: 1740 1691 672 606 1711 117 1 1559 173 
Answer: 874 1529 1745 1 89 1 1 586 17 1 1 
Answer: 1559 1 1 107 1 1119 69 1559 1327 765 6 1 
Answer: 1559 1 667 606 475 1007 1559 564 1525 999 1 946 
Answer: 1559 1 1 1 1 7 1 1585 269 
Answer: 1744 186 1 1 611 1559 1 1 707 767 1 1 
Answer: 1559 141 1689 1585 634 1565 765 179 1 1559 1 1585 1559 802 1 
Answer: 1559 249 1396 667 17 1 1 1 1 1 765 606 
Answer: 1571 165 999 1 667 1 1 69 765 1 
Answer: 1559 1298 1125 1 1 1 1 345 1717 17 1222 
Answer: 1559 1055 1 1 1559 1 765 1 1 1585 1 1 
Answer: 1740 1571 1 1 1 1 1438 1745 1 1713 1721 
Answer: 1559 1 1016 1 107 10 60 586 1 1 69 1 
Answer: 1 968 1585 1338 1745 1579 107 1559 624 1585 472 
Answer: 1559 1 1494 59 1 1 1 586 1 69 1 
Answer: 1744 186 184 1559 1 1 1585 221 1559 1 1 
Answer: 1559 1 667 17 1 586 1 1007 1 
Answer: 1559 824 1 1744 1585 1254 1745 1 1 
Answer: 1559 1 1 1 1 177 984 1 335 
Answer: 1559 1 1 1 1 1 69 1559 802 1010 1 107 4 1119 
Answer: 1585 1 1 692 1605 1622 1 1 11 1119 
Answer: 1571 1313 765 1 586 1 1 968 1585 1542 1559 1 
Answer: 1567 1382 17 1 908 107 1558 1 1 
Answer: 1559 1692 588 1 1 1 1588 1 
Answer: 1744 186 1072 1745 1653 1 1011 1025 107 1559 842 1001 
Answer: 1 17 1 1 707 1744 1 1559 1 740 999 1 1 
Answer: 1571 1157 765 267 1717 163 1 69 1715 1538 
token_cnt length: 15
vocab_size = 1748
Embedding matrix initialized with random values.
PositionalEncoding Constructor
Attention weights for layer 0 initialized with random values.
Attention weights for layer 1 initialized with random values.
Attention weights for layer 2 initialized with random values.
Attention weights for layer 3 initialized with random values.
Attention weights for layer 4 initialized with random values.
Attention weights for layer 5 initialized with random values.
Transformer initialized with 6 layers.
Do you want to start mini prompt mode? (Y/N): n
Continuing with training loop...
learning_rate: 1e-05
momentum: 0.9
Epoch 1 / 5000
** correct_prob : 0.49026
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
Average Loss for Epoch 1: 4.55118
Epoch 2 / 5000
** correct_prob : 0.548701
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
Average Loss for Epoch 2: 2.80552
Epoch 3 / 5000
** correct_prob : 0.542208
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
Average Loss for Epoch 3: 2.19552
Epoch 4 / 5000

```


