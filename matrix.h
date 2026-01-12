#pragma once

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <vector>
#include <string>
#include <sstream>
#include <set>
#include <map>
#include <cmath>
#include <future>
#include <thread>

// 平台和指令集检测宏
#if defined(__APPLE__) && defined(__MACH__)
    // Apple 平台：优先使用 Apple 的 simd/Accelerate 接口
    #ifndef SIMD_ARCH_APPLE_METAL
        #define SIMD_ARCH_APPLE_METAL
    #endif
    // 在 Apple arm64 上也可能支持 NEON intrinsics
    #if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
        #ifndef SIMD_ARCH_ARM_NEON
            #define SIMD_ARCH_ARM_NEON
        #endif
        #include <arm_neon.h>
    #endif
    #include <simd/simd.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    #ifndef SIMD_ARCH_ARM_NEON
        #define SIMD_ARCH_ARM_NEON
    #endif
    #include <arm_neon.h>
#elif defined(__AVX__) || defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    #ifndef SIMD_ARCH_X86_SSE
        #define SIMD_ARCH_X86_SSE
    #endif
    #include <xmmintrin.h>
    #ifdef __SSE2__
        #include <emmintrin.h>
    #endif
    #ifdef __SSE3__
        #include <pmmintrin.h>
    #endif
    #ifdef __SSE4_1__
        #include <smmintrin.h>
    #endif
    #ifdef __AVX__
        #include <immintrin.h>
    #endif
#else
    #define SIMD_ARCH_GENERIC
#endif

// 统一的SIMD类型定义
#if defined(SIMD_ARCH_ARM_NEON)
    typedef float32x4_t simd_f32;
    typedef float32x4x2_t simd_f32x2;
#elif defined(SIMD_ARCH_X86_SSE)
    #ifdef __AVX__
        typedef __m256 simd_f32;
        typedef struct { __m256 a, b; } simd_f32x2;
    #else
        typedef __m128 simd_f32;
        typedef struct { __m128 a, b; } simd_f32x2;
    #endif
#elif defined(SIMD_ARCH_APPLE_METAL)
    typedef simd::float4 simd_f32;
    typedef struct { simd::float4 a, b; } simd_f32x2;
#else
    // 通用回退：使用标量数组模拟
    typedef struct { float data[4]; } simd_f32;
    typedef struct { simd_f32 a, b; } simd_f32x2;
#endif

// 统一的SIMD操作函数
namespace simd_ops {
    // SIMD vector width in floats (4 for SSE/NEON, 8 for AVX)
#if defined(SIMD_ARCH_X86_SSE) && defined(__AVX__)
    static constexpr int SIMD_WIDTH = 8;
#else
    static constexpr int SIMD_WIDTH = 4;
#endif

    // 加载操作
    inline simd_f32 load(const float* ptr);
    
    // 设置标量到所有通道
    inline simd_f32 set1(float value);
    
    // 加法
    inline simd_f32 add(simd_f32 a, simd_f32 b);
    
    // 乘法
    inline simd_f32 mul(simd_f32 a, simd_f32 b);
    
    // 乘加：a * b + c
    inline simd_f32 fmadd(simd_f32 a, simd_f32 b, simd_f32 c);
    
    // 水平求和
    inline float horizontal_sum(simd_f32 v);
}

// =============================================================================
// 基础矩阵操作函数
// =============================================================================

// 内存对齐分配辅助函数
void* aligned_alloc_helper(size_t align, size_t size);

// 向量点积
float vec_dot(const float* x, const float* y, int n);

// 基础矩阵乘法（三重循环）
float* base_mul(const float* A, const float* B, float* C, int r, int k, int c, int bs = 0);

// 矩阵转置乘法
float* matrix_mul_trans(const float* A, const float* B, float* res, int r, int k, int c, int bs = 0);

// 分块矩阵乘法
float* matrix_mul_block(const float* A, const float* B, float* res, int r, int k, int c, int block_size = 64);

// 分块转置矩阵乘法
float* matrix_mul_trans_block(const float* A, const float* B, float* res, int r, int k, int c, int block_size = 64);

// SIMD优化的分块转置矩阵乘法
float* matrix_mul_trans_block_with_simd(const float* A, const float* B, float* res, int r, int k, int c, int block_size = 64);

// 异步SIMD优化的分块转置矩阵乘法（多线程）
float* async_matrix_mul_trans_block_with_simd(const float* A, const float* B, float* res, int r, int k, int c, int block_size = 64, int num_threads = -1);

// =============================================================================
// 平台特定优化函数
// =============================================================================

#ifdef SIMD_ARCH_X86_SSE
// AVX微内核（6x16）
void avx_micro_kernel_6x16(const float* A, const float* B, float* C, 
                           int i, int j, int p, int i_end, int j_end, int p_end,
                           int k, int ldc);

// Xeon E5优化版本
float* optimized_matrix_mul_xeon_e5(const float* A, const float* B, 
                                  float* res, int r, int k, int c, int block_size = 64);
#endif

#ifdef SIMD_ARCH_APPLE_METAL
// Apple M2优化版本（使用Accelerate框架）
float* optimized_matrix_mul_apple_m2(const float* A, const float* B, float* C, int r, int k, int c, int bs = 0);

// 自定义NEON优化版本
float* my_optimized_matrix_mul(const float* A, const float* B, float* res, int r, int k, int c, int block_size = 144);
#endif

#if defined(SIMD_ARCH_ARM_NEON) || defined(SIMD_ARCH_APPLE_METAL)
// ARM NEON优化版本
float* optimized_matrix_mul_arm_neon(const float* A, const float* B, float* C, int r, int k, int c);

// NEON + OpenMP优化版本
float* optimized_matrix_mul_neon_omp(const float* A, const float* B, float* C, int r, int k, int c, int block_size = 128);

// NEON微内核函数
static inline void neon_micro_kernel_4x8(const float* A_block, const float* B_block, float* C_block, 
                                         int k, int ldc, int prefetch_offset = 64);
static inline void neon_micro_kernel_4x4(const float* A_block, const float* B_block, float* C_block, int k, int ldc);

// NEON辅助函数
inline float horizontal_sum_neon(float32x4_t v);
inline void load_a_block_broadcast(float32x4_t a_vec[], const float* A, int i, int k, int p, int p_end);
inline void load_b_block(float32x4_t b_vec[], const float* B_transposed, int j, int k, int p, int p_end);
inline void load_c_block(float32x4_t c_regs[][2], const float* C, int i, int j, int ldc);
inline void store_c_block(const float32x4_t c_regs[][2], float* C, int i, int j, int ldc);
inline void neon_outer_product_update(float32x4_t c_regs[][2], float32x4_t a_vec, float32x4_t b_vec[]);
void aggressive_neon_kernel(const float* A, const float* B_transposed, float* C, int r, int k, int c);
#endif

// =============================================================================
// 通用优化函数
// =============================================================================

// 缓存优化的矩阵转置
void cache_optimized_transpose(float* dst, const float* src, int rows, int cols);

// 自适应最佳性能函数
float* best_matrix_mul(const float* A, const float* B, float* C, int r, int k, int c, int bs = 0);

// =============================================================================
// 工具函数
// =============================================================================

// 随机数生成
float rand_float(float s);

// 生成随机矩阵
float* random_matrix(int r, int c, float seed);

// 生成测试矩阵
void matrix_gen(float *a, float *b, int N, float seed);

// 计算矩阵迹
float Trace(const float* A, int r, int c);

// 打印矩阵
void print_matrix(const float* A, int r, int c);

// 矩阵比较
bool comp(float *a, float *b, int N);

// =============================================================================
// 测试和解析函数
// =============================================================================

// 打印帮助信息
inline void print_help();

// 解析方法列表
std::set<std::string> parse_methods(const std::string& method_list);

// 验证方法有效性
bool is_valid_method(const std::string& method);

// 测试模块
void test_mod(int argc, char** argv);