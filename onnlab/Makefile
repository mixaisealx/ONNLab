# Compiler and flags
CXX := g++
CXXFLAGS := -std=c++20 -m32 -O3

# Directories locations
SRC_DIR := .
OBJ_DIR := obj
BIN_DIR := ../bin

# Output binary location
TARGET := $(BIN_DIR)/experiment

# Find all cpp files and create corresponding object files
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

# Default target
all: $(TARGET)

# Link object files into final binary
$(TARGET): $(OBJS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile .cpp files to .o files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -rf $(OBJ_DIR) $(TARGET)

.PHONY: all clean