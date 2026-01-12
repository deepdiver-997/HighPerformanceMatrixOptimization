# Matrix Library

这个库包含了高性能矩阵乘法函数的实现，支持多种SIMD指令集和优化技术。

## 文件结构

- `matrix.h` - 包含所有函数声明和数据结构定义
- `matrix.cpp` - 包含所有函数实现
- `main_test.cpp` - 示例主程序，演示如何使用库

## 支持的平台和指令集

### ARM 平台
- **ARM NEON** - 基础NEON SIMD指令
- **Apple Metal** - Apple Silicon优化的SIMD指令

### x86 平台  
- **SSE** - 基础SIMD指令集
- **AVX** - 256位向量指令集
- **AVX2/AVX-512** - 更高级的向量指令集

## 主要功能

### 基础矩阵操作
```cpp
// 基础三重循环矩阵乘法
float* base_mul(const float* A, const float* B, float* C, int r, int k, int c);

// 分块矩阵乘法
float* matrix_mul_block(const float* A, const float* B, float* C, int r, int k, int c, int block_size = 64);

// 矩阵转置乘法
float* matrix_mul_trans(const float* A, const float* B, float* C, int r, int k, int c);

// 分块转置矩阵乘法
float* matrix_mul_trans_block(const float* A, const float* B, float* C, int r, int k, int c, int block_size = 64);
```

### SIMD优化版本
```cpp
// SIMD优化的分块转置矩阵乘法
float* matrix_mul_trans_block_with_simd(const float* A, const float* B, float* C, int r, int k, int c, int block_size = 64);

// 异步多线程SIMD优化版本
float* async_matrix_mul_trans_block_with_simd(const float* A, const float* B, float* C, int r, int k, int c, int block_size = 64, int num_threads = -1);
```

### 平台特定优化
```cpp
#ifdef SIMD_ARCH_APPLE_METAL
// Apple M2 Pro优化版本
float* optimized_matrix_mul_apple_m2(const float* A, const float* B, float* C, int r, int k, int c);
float* my_optimized_matrix_mul(const float* A, const float* B, float* C, int r, int k, int c);
#endif

#ifdef SIMD_ARCH_X86_SSE
// Intel Xeon优化版本
float* optimized_matrix_mul_xeon_e5(const float* A, const float* B, float* C, int r, int k, int c);
#endif

#if defined(SIMD_ARCH_ARM_NEON) || defined(SIMD_ARCH_APPLE_METAL)
// ARM NEON优化版本
float* optimized_matrix_mul_arm_neon(const float* A, const float* B, float* C, int r, int k, int c);
float* optimized_matrix_mul_neon_omp(const float* A, const float* B, float* C, int r, int k, int c);
#endif
```

### 自适应最佳性能
```cpp
// 自动选择当前平台最优化的实现
float* best_matrix_mul(const float* A, const float* B, float* C, int r, int k, int c);
```

## 工具函数

```cpp
// 内存分配
void* aligned_alloc_helper(size_t align, size_t size);

// 矩阵生成和操作
void matrix_gen(float* A, float* B, int N, float seed);
float Trace(const float* A, int r, int c);
void print_matrix(const float* A, int r, int c);
bool comp(float* A, float* B, int N);

// 性能测试
void test_mod(int argc, char** argv);
```

## 编译方法

### Apple Silicon (M2 Pro)
```bash
/opt/homebrew/opt/llvm/bin/clang++ -O3 -march=native -fopenmp -std=c++17 -DSIMD_ARCH_APPLE_METAL -framework Accelerate main_test.cpp matrix.cpp -o matrix_test
```

### Intel x86 平台
```bash
clang++ -O3 -march=native -fopenmp -std=c++17 main_test.cpp matrix.cpp -o matrix_test
```

### ARM Linux 平台
```bash
clang++ -O3 -march=native -fopenmp -std=c++17 main_test.cpp matrix.cpp -o matrix_test
```

## 使用示例

### 基本使用
```cpp
#include "matrix.h"

int main() {
    const int N = 1024;
    float *A = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
    float *B = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
    float *C = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
    
    // 生成测试矩阵
    matrix_gen(A, B, N, 0.3f);
    
    // 使用SIMD优化版本进行矩阵乘法
    float* result = async_matrix_mul_trans_block_with_simd(A, B, C, N, N, N, 64, 4);
    
    // 计算性能
    float trace = Trace(result, N, N);
    std::cout << "Trace: " << trace << std::endl;
    
    // 释放内存
    free(A);
    free(B);
    free(C);
    
    return 0;
}
```

### 性能测试
```bash
# 运行完整性能测试
./matrix_test --test --sizes 512,1024,2048,4096 --threads 8 --block-size 128 --times 3

# 查看测试选项
./matrix_test --test --help
```

## 测试选项

`test_mod`函数支持以下命令行参数：

- `-t, --threads <num>`: 设置线程数（默认：硬件并发数）
- `-b, --block-size <size>`: 设置分块大小（默认：64）
- `-s, --sizes <list>`: 设置矩阵大小列表（逗号分隔，默认：512,1024,2048,4096）
- `-e, --seed <value>`: 设置随机种子（默认：0.3）
- `-n, --times <num>`: 设置每个测试的运行次数（默认：3）
- `-h, --help`: 显示帮助信息

## 性能特征

### 支持的优化技术
1. **矩阵转置** - 优化内存访问模式
2. **分块计算** - 适应缓存层次结构
3. **SIMD向量化** - 利用单指令多数据并行
4. **多线程并行** - 利用多核处理器
5. **数据预取** - 减少缓存缺失延迟

### 典型性能 (Apple M2 Pro)
- 512×512 矩阵: ~85 GFlops
- 1024×1024 矩阵: ~80 GFlops  
- 2048×2048 矩阵: ~68 GFlops
- 4096×4096 矩阵: ~144 GFlops (8线程)

## 依赖项

- C++17 或更高版本的编译器
- OpenMP 支持用于多线程
- Apple平台需要Accelerate框架
- 线程池实现 (pool.h, pool.cpp)

## 注意事项

1. 所有矩阵数据都是64字节对齐的，以获得最佳SIMD性能
2. 多线程版本使用线程池，避免频繁创建销毁线程
3. 函数会自动检测并使用当前平台最优化的SIMD指令集
4. 内存分配失败时函数会返回nullptr，调用者需要检查返回值