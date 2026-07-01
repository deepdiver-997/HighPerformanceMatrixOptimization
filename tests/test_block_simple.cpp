#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <cstring>
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
namespace simd_ops {
    static constexpr int SIMD_WIDTH = 4;
    // 加载操作
    inline simd_f32 load(const float* ptr) {
#if defined(SIMD_ARCH_ARM_NEON)
        return vld1q_f32(ptr);
#elif defined(SIMD_ARCH_X86_SSE)
        #ifdef __AVX__
            return _mm256_loadu_ps(ptr);
        #else
            return _mm_loadu_ps(ptr);
        #endif
#elif defined(SIMD_ARCH_APPLE_METAL)
        return simd_make_float4(ptr[0], ptr[1], ptr[2], ptr[3]);
#else
        simd_f32 result;
        for (int i = 0; i < 4; ++i) result.data[i] = ptr[i];
        return result;
#endif
    }

    // 设置标量到所有通道
    inline simd_f32 set1(float value) {
#if defined(SIMD_ARCH_ARM_NEON)
        return vdupq_n_f32(value);
#elif defined(SIMD_ARCH_X86_SSE)
        #ifdef __AVX__
            return _mm256_set1_ps(value);
        #else
            return _mm_set1_ps(value);
        #endif
#elif defined(SIMD_ARCH_APPLE_METAL)
        return simd::float4(value);
#else
        simd_f32 result;
        for (int i = 0; i < 4; ++i) result.data[i] = value;
        return result;
#endif
    }

    // 加法
    inline simd_f32 add(simd_f32 a, simd_f32 b) {
#if defined(SIMD_ARCH_ARM_NEON)
        return vaddq_f32(a, b);
#elif defined(SIMD_ARCH_X86_SSE)
        #ifdef __AVX__
            return _mm256_add_ps(a, b);
        #else
            return _mm_add_ps(a, b);
        #endif
#elif defined(SIMD_ARCH_APPLE_METAL)
        return a + b;
#else
        simd_f32 result;
        for (int i = 0; i < 4; ++i) result.data[i] = a.data[i] + b.data[i];
        return result;
#endif
    }

    // 乘法
    inline simd_f32 mul(simd_f32 a, simd_f32 b) {
#if defined(SIMD_ARCH_ARM_NEON)
        return vmulq_f32(a, b);
#elif defined(SIMD_ARCH_X86_SSE)
        #ifdef __AVX__
            return _mm256_mul_ps(a, b);
        #else
            return _mm_mul_ps(a, b);
        #endif
#elif defined(SIMD_ARCH_APPLE_METAL)
        return a * b;
#else
        simd_f32 result;
        for (int i = 0; i < 4; ++i) result.data[i] = a.data[i] * b.data[i];
        return result;
#endif
    }

    // 乘加：a * b + c
    inline simd_f32 fmadd(simd_f32 a, simd_f32 b, simd_f32 c) {
#if defined(SIMD_ARCH_ARM_NEON)
        return vmlaq_f32(c, a, b);
#elif defined(SIMD_ARCH_X86_SSE) && defined(__FMA__)
        return _mm_fmadd_ps(a, b, c);
#elif defined(SIMD_ARCH_X86_SSE)
        return add(mul(a, b), c);
#elif defined(SIMD_ARCH_APPLE_METAL)
        return a * b + c;
#else
        simd_f32 result;
        for (int i = 0; i < 4; ++i) result.data[i] = a.data[i] * b.data[i] + c.data[i];
        return result;
#endif
    }

    // 水平求和
    inline float horizontal_sum(simd_f32 v) {
#if defined(SIMD_ARCH_ARM_NEON)
    // on aarch64 vaddvq_f32 performs a vector-wide horizontal add
#if defined(__aarch64__)
    return vaddvq_f32(v);
#else
    float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(sum, sum), 0);
#endif
#elif defined(SIMD_ARCH_X86_SSE)
    #ifdef __AVX__
        // AVX version with 8 floats
        __m128 sum128 = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
    #else
        // SSE version with 4 floats
        __m128 sum = _mm_hadd_ps(v, v);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    #endif
#elif defined(SIMD_ARCH_APPLE_METAL)
        return simd::reduce_add(v);
#else
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) sum += v.data[i];
        return sum;
#endif
    }
}

// 简化的内存对齐分配
void* aligned_alloc_helper(size_t align, size_t size) {
    void* ptr = nullptr;
    size_t sz = ((size + align - 1) / align) * align;
    if (posix_memalign(&ptr, align, sz) != 0) return nullptr;
    return ptr;
}

// 基础矩阵转置分块乘法
float* matrix_mul_trans_block(const float* A, const float* B, float* res, int r, int k, int c, int bs) {
    const size_t align = 64;
    float *b = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * c * k));
    if (!res)
        res = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    if (!b || !res) {
        std::cerr << "Aligned allocation failed" << std::endl;
        free(b);
        free(res);
        return nullptr;
    }
    memset(res, 0, sizeof(float) * r * c);
    
    // 转置B矩阵
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < k; ++j) {
            b[i * k + j] = B[j * c + i];
        }
    }
    
    // 分块计算
    for (int i = 0; i < r; i += bs) {
        int i_end = std::min(i + bs, r);
        for (int j = 0; j < c; j += bs) {
            int j_end = std::min(j + bs, c);
            for (int p = 0; p < k; p += bs) {
                int p_end = std::min(p + bs, k);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = 0.0f;
                        for (int pp = p; pp < p_end; ++pp) {
                            sum += A[ii * k + pp] * b[jj * k + pp];
                        }
                        res[ii * c + jj] += sum;
                    }
                }
            }
        }
    }
    
    free(b);
    return res;
}

// SIMD优化的分块转置矩阵乘法
float* matrix_mul_trans_block_with_simd(const float* A, const float* B, float* res, int r, int k, int c, int block_size) {
    const size_t align = 64;
    float *b = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * c * k));
    if (!res)
        res = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    if (!b || !res) {
        std::cerr << "Aligned allocation failed" << std::endl;
        free(b);
        free(res);
        return nullptr;
    }
    memset(res, 0, sizeof(float) * r * c);
    for (int i = 0; i < c; ++i) {
        for(int j = 0; j < k; ++j) {
            b[i * k + j] = B[j * c + i];
        }
    }
    
    for (int i = 0; i < r; i += block_size) {
        for (int j = 0; j < c; j += block_size) {
            for (int p = 0; p < k; p += block_size) {
                int i_end = std::min(i + block_size, r);
                int j_end = std::min(j + block_size, c);
                int p_end = std::min(p + block_size, k);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = 0.0f;
                        int pp;
                        
                        // 使用统一的SIMD类型和操作
                        simd_f32 sum_vec0 = simd_ops::set1(0.0f), sum_vec1 = simd_ops::set1(0.0f),
                                  sum_vec2 = simd_ops::set1(0.0f), sum_vec3 = simd_ops::set1(0.0f);
                        for (pp = p; pp + 3 * simd_ops::SIMD_WIDTH < p_end; pp += 4 * simd_ops::SIMD_WIDTH) {
                            simd_f32 a0 = simd_ops::load(&A[ii * k + pp]);
                            simd_f32 b0 = simd_ops::load(&b[jj * k + pp]);
                            sum_vec0 = simd_ops::fmadd(a0, b0, sum_vec0);

                            simd_f32 a1 = simd_ops::load(&A[ii * k + pp + simd_ops::SIMD_WIDTH]);
                            simd_f32 b1 = simd_ops::load(&b[jj * k + pp + simd_ops::SIMD_WIDTH]);
                            sum_vec1 = simd_ops::fmadd(a1, b1, sum_vec1);

                            simd_f32 a2 = simd_ops::load(&A[ii * k + pp + 2 * simd_ops::SIMD_WIDTH]);
                            simd_f32 b2 = simd_ops::load(&b[jj * k + pp + 2 * simd_ops::SIMD_WIDTH]);
                            sum_vec2 = simd_ops::fmadd(a2, b2, sum_vec2);

                            simd_f32 a3 = simd_ops::load(&A[ii * k + pp + 3 * simd_ops::SIMD_WIDTH]);
                            simd_f32 b3 = simd_ops::load(&b[jj * k + pp + 3 * simd_ops::SIMD_WIDTH]);
                            sum_vec3 = simd_ops::fmadd(a3, b3, sum_vec3);
                        }
                        
                        // 处理剩余的元素
                        for (; pp < p_end; ++pp) {
                            sum += A[ii * k + pp] * b[jj * k + pp];
                        }
                        
                        // 将SIMD结果加到标量和中
                        sum += simd_ops::horizontal_sum(sum_vec0);
                        sum += simd_ops::horizontal_sum(sum_vec1);
                        sum += simd_ops::horizontal_sum(sum_vec2);
                        sum += simd_ops::horizontal_sum(sum_vec3);
                        res[ii * c + jj] += sum;
                    }
                }
            }
        }
    }
    
    free(b);
    return res;
}

// 测试不同分块大小的性能
void test_block_sizes(int matrix_size, const std::vector<int>& block_sizes, int test_times = 3) {
    std::cout << "测试矩阵大小: " << matrix_size << "x" << matrix_size << std::endl;
    std::cout << std::setw(8) << "分块大小" << std::setw(12) << "平均时间(ms)" << std::setw(12) << "最小时间(ms)" << std::setw(12) << "最大时间(ms)" << std::endl;
    std::cout << std::string(44, '-') << std::endl;
    
    // 创建测试矩阵
    float* A = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * matrix_size * matrix_size));
    float* B = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * matrix_size * matrix_size));
    
    // 初始化随机矩阵
    srand(42);
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    srand(123);
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    std::vector<std::pair<int, double>> results;
    
    for (int block_size : block_sizes) {
        std::vector<double> times;
        
        for (int t = 0; t < test_times; ++t) {
            auto start = std::chrono::high_resolution_clock::now();
            float* C = matrix_mul_trans_block(A, B, nullptr, matrix_size, matrix_size, matrix_size, block_size);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0); // 转换为毫秒
            free(C);
        }
        
        double avg_time = 0;
        double min_time = times[0];
        double max_time = times[0];
        
        for (double time : times) {
            avg_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        avg_time /= test_times;
        
        std::cout << std::setw(8) << block_size 
                  << std::setw(12) << std::fixed << std::setprecision(2) << avg_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << min_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << max_time << std::endl;
        
        results.push_back({block_size, avg_time});
    }
    
    // 找出最优分块大小
    auto best = std::min_element(results.begin(), results.end(),
        [](const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.second < b.second; });
    
    std::cout << "\n最优分块大小: " << best->first << " (平均时间: " << best->second << " ms)" << std::endl;
    
    // 使用最优分块大小测试SIMD版本
    std::cout << "\n使用最优分块大小测试SIMD版本..." << std::endl;
    
    std::vector<double> simd_times;
    for (int t = 0; t < test_times; ++t) {
        auto start = std::chrono::high_resolution_clock::now();
        float* C = matrix_mul_trans_block_with_simd(A, B, nullptr, matrix_size, matrix_size, matrix_size, best->first);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        simd_times.push_back(duration.count() / 1000.0);
        free(C);
    }
    
    double simd_avg = 0, simd_min = simd_times[0], simd_max = simd_times[0];
    for (double time : simd_times) {
        simd_avg += time;
        simd_min = std::min(simd_min, time);
        simd_max = std::max(simd_max, time);
    }
    simd_avg /= test_times;
    
    std::cout << "SIMD版本性能:" << std::endl;
    std::cout << "  平均时间: " << std::fixed << std::setprecision(2) << simd_avg << " ms" << std::endl;
    std::cout << "  最小时间: " << std::fixed << std::setprecision(2) << simd_min << " ms" << std::endl;
    std::cout << "  最大时间: " << std::fixed << std::setprecision(2) << simd_max << " ms" << std::endl;
    
    double speedup = best->second / simd_avg;
    std::cout << "  加速比: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    // 清理内存
    free(A);
    free(B);
}

int main() {
    std::cout << "=== 矩阵分块性能测试 ===" << std::endl;
    
    // 测试不同矩阵大小
    std::vector<int> matrix_sizes = {4096};
    std::vector<int> block_sizes = {32};
    
    for (int size : matrix_sizes) {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        test_block_sizes(size, block_sizes, 3);
    }
    
    return 0;
}