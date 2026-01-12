# Makefile for compiling m2.cpp on Xeon E5-2609 v2 (Ivy Bridge) server
# Optimized for AVX, OpenMP, and high-performance matrix multiplication

# Compiler
CC = g++

# Compiler flags for maximum optimization on Ivy Bridge
CFLAGS = -O3 \
         -march=ivybridge \
         -mtune=ivybridge \
         -fopenmp \
         -ffast-math \
         -funroll-loops \
         -mprefer-vector-width=256 \
         -flto \
         -std=c++17 \
         -pthread \
         -DNDEBUG \
         -Wno-unused-variable \
         -Wno-unused-but-set-variable

# Linker flags
LDFLAGS = -fopenmp -flto

# Target executable
TARGET = m2

# Source files
SRCS = m2.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Default target
all: $(TARGET)

# Link the executable
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) -o $@ $^

# Compile source files to object files
%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up build artifacts
clean:
	rm -f $(OBJS) $(TARGET)

# Phony targets
.PHONY: all clean