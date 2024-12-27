# mini_transformer_cpp
Minimal transformer in C/C++ no dependancy of machine learning library just plain C/C++ code
Just for my own tutorial proposal
## File structures
```
transformer_project/
├── src/
│   ├── main.cpp            # Entry point for the application
│   ├── transformer.cpp     # Implementation of the Transformer
│   ├── attention.cpp       # Implementation of the attention mechanism
│   ├── feed_forward.cpp    # Implementation of the feed-forward network
│   ├── embedding.cpp       # Implementation of embeddings
│   ├── positional_encoding.cpp # Implementation of positional encoding
│   └── utils.cpp           # Helper functions
├── include/
│   ├── transformer.h
│   ├── attention.h
│   ├── feed_forward.h
│   ├── embedding.h
│   ├── positional_encoding.h
│   └── utils.h
├── build/                  # Directory for compiled files
├── data/                   # Sample input data for testing
└── Makefile                # Build instructions
```
## Understand position encoder
Purpose of Positional Encoding

The purpose of the positional encoding is to inject information about the position of each token in the sequence. Transformers process sequences in parallel without inherent knowledge of token order, so positional encoding compensates for this.

    Key Idea:
        pos_encoding[pos][i] stores the positional encoding for the i-th dimension of the pos-th token.
        Sine (sin) and Cosine (cos) functions with varying frequencies are used to encode positions.
### Plot the sin and cos initilized vector
Use the octave to plot sin.dat file
esample data:
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


