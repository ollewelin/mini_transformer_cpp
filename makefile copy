# Compiler and flags
CC = g++ -std=c++14 -O3
CFLAGS = -g -Wall

# Directories
SRC_DIR = src
BUILD_DIR = build
INCLUDE_DIR = include

# Source and object files
SRCS = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS))

# Output executable
TARGET = transformer_app

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

# Object file compilation
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Clean build files
clean:
	rm -rf $(BUILD_DIR) $(TARGET)

# Rebuild everything
rebuild: clean all

# Phony targets
.PHONY: all clean rebuild
