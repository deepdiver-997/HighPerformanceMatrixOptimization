# =============================================================================
# Makefile for Metal GPU Matrix Multiplication
#
# 文件结构:
#   metal/
#   ├── metal_shaders.metal     GPU kernel 源码
#   ├── metal_matmul.h          C++ 接口头文件
#   ├── metal_matmul.mm         ObjC++ Metal 实现
#   ├── test_metal.cpp          C++ 测试程序
#   ├── METAL_GUIDE.md          Metal 学习指南
#   └── Makefile                本文件
#   ../
#   ├── matrix.cpp              CPU 矩阵运算库
#   ├── matrix.h                矩阵运算头文件
#   ├── pool.h                  线程池
#   └── getCacheSize.h          缓存检测
#
# 使用方法:
#   cd metal && make          # 编译并运行
#   cd metal && make build    # 只编译
#   cd metal && make run      # 运行测试
#   cd metal && make clean    # 清理
#   cd metal && make quick    # 快速小矩阵测试
#
#   make run S=2048           # 测试指定矩阵大小
#   make run S=4096 N=5       # 自定义大小和运行次数
#
# 依赖:
#   - macOS with Metal support
#   - Xcode Command Line Tools
# =============================================================================

CXX = clang++

# C++ 编译选项 (.cpp 文件)
CPPFLAGS = -std=c++17 -O3 -ffast-math -funroll-loops -pthread -DNDEBUG

# ObjC++ 编译选项 (.mm 文件)
# -fobjc-arc: 启用 ARC, -Wno-deprecated-declarations: 忽略 shader 运行时编译警告
MMFLAGS = -std=c++17 -O3 -fobjc-arc -Wno-deprecated-declarations

# 链接选项: Metal.framework + Foundation.framework
LDFLAGS = -framework Metal -framework Foundation -pthread

# 源文件 (matrix.cpp 在父目录)
MM_SRCS  = metal_matmul.mm
CPP_SRCS = test_metal.cpp ../matrix.cpp

MM_OBJS  = $(MM_SRCS:.mm=.o)
CPP_OBJS = $(CPP_SRCS:.cpp=.o)

TARGET = test_metal
SHADER = metal_shaders.metal

# =============================================================================
.PHONY: all build run clean quick test help

all: build run

build: $(TARGET)

$(TARGET): $(MM_OBJS) $(CPP_OBJS)
	$(CXX) -o $@ $^ $(LDFLAGS)
	@echo "Build: $(TARGET) ready"

# ObjC++ 编译
%.o: %.mm metal_matmul.h
	$(CXX) $(MMFLAGS) -c $< -o $@

# C++ 编译 (本地文件和 ../ 文件)
%.o: %.cpp
	$(CXX) $(CPPFLAGS) -c $< -o $@

# 显式依赖声明
metal_matmul.o: metal_matmul.mm metal_matmul.h
test_metal.o:    test_metal.cpp metal_matmul.h ../matrix.h ../pool.h ../getCacheSize.h
../matrix.o:     ../matrix.cpp ../matrix.h ../pool.h ../getCacheSize.h

run: $(TARGET) $(SHADER)
	@echo "=== Metal GPU Matrix Multiplication Benchmark ==="
	@echo ""
	./$(TARGET) $(if $(S),-s $(S)) $(if $(N),-n $(N))

# 快速测试：只跑 512x512，1次
quick: $(TARGET)
	./$(TARGET) -s 512 -n 1

clean:
	rm -f *.o ../matrix.o $(TARGET)
	@echo "Cleaned"

# 一步测试：清理 → 编译 → 快速测试 → 清理
test: clean build quick clean
	@echo "Test passed"

help:
	@echo "Metal GPU Matrix Multiplication - Build & Test"
	@echo ""
	@echo "Targets:"
	@echo "  make          Build and run full benchmark"
	@echo "  make build    Compile only"
	@echo "  make run      Run benchmark (after build)"
	@echo "  make quick    Quick test (512x512, 1 run)"
	@echo "  make clean    Remove build artifacts"
	@echo "  make test     Clean build + quick test + clean"
	@echo ""
	@echo "Variables:"
	@echo "  S=<size>     Matrix size (default: 256,512,1024,2048)"
	@echo "  N=<runs>     Runs per test (default: 3)"
	@echo ""
	@echo "Examples:"
	@echo "  make run S=2048"
	@echo "  make run S=4096 N=5"
