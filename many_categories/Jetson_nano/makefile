CXX = g++
CXXFLAGS = -g -Wall -Iinclude -std=c++17
LDFLAGS = -lstdc++fs  # Linker flags

TARGET = transformer_app
SRCS = src/main.cpp src/attention.cpp src/embedding.cpp src/feed_forward.cpp \
       src/positional_encoding.cpp src/transformer.cpp src/utils.cpp \
       src/config.cpp src/dataset.cpp src/layer_normalization.cpp \
       src/simple_normalizer.cpp
OBJS = $(patsubst src/%.cpp,build/%.o,$(SRCS))

all: build/$(TARGET)

build/$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)  # Linker flags applied only here

build/%.o: src/%.cpp
	mkdir -p build  # Create the 'build' directory at the top level
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf build
.PHONY: all clean
