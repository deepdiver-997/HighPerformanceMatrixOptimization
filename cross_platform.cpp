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
// #include <openmp_wrappers/math.h>

#include "getCacheSize.h"
#include "pool.h"

// aligned allocation helper
static void* aligned_alloc_helper(size_t align, size_t size) {
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    // aligned_alloc requires size to be multiple of align
    size_t sz = ((size + align - 1) / align) * align;
    return aligned_alloc(align, sz);
#else
    void* ptr = nullptr;
    size_t sz = ((size + align - 1) / align) * align;
    if (posix_memalign(&ptr, align, sz) != 0) return nullptr;
    return ptr;
#endif
}

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
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
#endif
#elif defined(SIMD_ARCH_X86_SSE)
        #ifdef __AVX__
            __m128 sum128 = _mm_add_ps(_mm256_extractf128_ps(v, 1), _mm256_castps256_ps128(v));
            sum128 = _mm_hadd_ps(sum128, sum128);
            sum128 = _mm_hadd_ps(sum128, sum128);
            return _mm_cvtss_f32(sum128);
        #else
            __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(v, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        #endif
#elif defined(SIMD_ARCH_APPLE_METAL)
        return v[0] + v[1] + v[2] + v[3];
#else
        return v.data[0] + v.data[1] + v.data[2] + v.data[3];
#endif
    }
}

struct timer {
    std::chrono::steady_clock::time_point start, end;
    timer() { start = std::chrono::steady_clock::now(); }
    ~timer() {
        // end = std::chrono::steady_clock::now();
        // std::cout << "Elapsed time: " 
        //           << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
        //           << " ms" << std::endl;
    }
    void start_timer() { start = std::chrono::steady_clock::now(); }
    void end_timer() { 
        end = std::chrono::steady_clock::now();
        std::cout << "Elapsed time: " 
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() 
                  << " ms" << std::endl;
    }
};

float vec_dot(const float* x, const float* y, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += x[i] * y[i];
    }
    return sum;
}

constexpr int block_sizes[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};

float* matrix_mul_trans(const float* A, const float* B, float* res, int r, int k, int c, int bs = 0) {
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
    
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            res[i * c + j] = vec_dot(&A[i * k], &b[j * k], k);
        }
    }
    
    free(b);
    return res;
}

float* base_mul(const float* A, const float* B, float* C, int r, int k, int c, int bs = 0) {
    const size_t align = 64;
    if (!C)
        C = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    if (!C) {
        std::cerr << "Aligned allocation failed" << std::endl;
        return nullptr;
    }
    memset(C, 0, sizeof(float) * r * c);
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            float tmp = 0.0f;
            for (int p = 0; p < k; ++p) {
                tmp += A[i * k + p] * B[p * c + j];
            }
            C[i * c + j] = tmp;
        }
    }
    return C;
}

float* matrix_mul_block(const float* A, const float* B,float *res, int r, int k, int c, int block_size = 64) {
    const size_t align = 64;
    if (!res)
        res = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    if (!res) return nullptr;
    memset(res, 0, sizeof(float) * r * c);
    for (int i = 0; i < r; i += block_size) {
        for (int j = 0; j < c; j += block_size) {
            for (int p = 0; p < k; p += block_size) {
                int i_end = std::min(i + block_size, r);
                int j_end = std::min(j + block_size, c);
                int p_end = std::min(p + block_size, k);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = 0.0f;
                        for (int pp = p; pp < p_end; ++pp) {
                            sum += A[ii * k + pp] * B[pp * c + jj];
                        }
                        res[ii * c + jj] += sum;
                    }
                }
            }
        }
    }
    return res;
}

float* matrix_mul_trans_block(const float* A, const float* B, float* res, int r, int k, int c, int block_size = 64) {
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

float* matrix_mul_trans_block_with_simd(const float* A, const float* B, float* res, int r, int k, int c, int block_size = 64) {
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

// float* async_matrix_mul_trans_block_with_simd(const float* A, const float* B, float* res, int r, int k, int c, int block_size = 64, int num_threads = -1) {
//     const size_t align = 64;
//     // Transpose B into b as other functions do
//     float *b = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * c * k));
//     if (!res)
//         res = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
//     if (!b || !res) {
//         std::cerr << "Aligned allocation failed" << std::endl;
//         free(b);
//         free(res);
//         return nullptr;
//     }
//     memset(res, 0, sizeof(float) * r * c);
//     for (int i = 0; i < c; ++i) {
//         for (int j = 0; j < k; ++j) {
//             b[i * k + j] = B[j * c + i];
//         }
//     }

//     // Determine number of worker threads to use
//     unsigned int hw = std::thread::hardware_concurrency();
//     int num_threads = hw == 0 ? 4 : static_cast<int>(hw);

//     // Create thread pool with same number of threads
//     if (!pool)
//         pool = std::make_shared<ThreadPool>(static_cast<size_t>(num_threads));

//     // Split work by rows: each task handles a contiguous range of rows [row_start, row_end)
//     int rows_per_task = (r + num_threads - 1) / num_threads;
//     std::vector<std::future<void>> futures;
//     futures.reserve(num_threads);

//     for (int t = 0; t < num_threads; ++t) {
//         int row_start = t * rows_per_task;
//         if (row_start >= r) break;
//         int row_end = std::min(r, row_start + rows_per_task);

//         // capture by value the pointers and ranges to avoid race on locals
//         auto fut = pool->enqueue_task([=, &A, &b, &res]() {
//             // For the assigned row range, perform blocked, transposed, simd-accelerated multiply
//             for (int i = row_start; i < row_end; i += block_size) {
//                 for (int j = 0; j < c; j += block_size) {
//                     for (int p = 0; p < k; p += block_size) {
//                         int i_end = std::min(i + block_size, row_end);
//                         int j_end = std::min(j + block_size, c);
//                         int p_end = std::min(p + block_size, k);

//                         for (int ii = i; ii < i_end; ++ii) {
//                             for (int jj = j; jj < j_end; ++jj) {
//                                 float sum = 0.0f;
//                                 int pp = p;

//                                 simd_f32 sum_vec0 = simd_ops::set1(0.0f), sum_vec1 = simd_ops::set1(0.0f),
//                                           sum_vec2 = simd_ops::set1(0.0f), sum_vec3 = simd_ops::set1(0.0f);
//                                 for (pp = p; pp + 3 * simd_ops::SIMD_WIDTH < p_end; pp += 4 * simd_ops::SIMD_WIDTH) {
//                                     simd_f32 a0 = simd_ops::load(&A[ii * k + pp]);
//                                     simd_f32 b0 = simd_ops::load(&b[jj * k + pp]);
//                                     sum_vec0 = simd_ops::fmadd(a0, b0, sum_vec0);

//                                     simd_f32 a1 = simd_ops::load(&A[ii * k + pp + simd_ops::SIMD_WIDTH]);
//                                     simd_f32 b1 = simd_ops::load(&b[jj * k + pp + simd_ops::SIMD_WIDTH]);
//                                     sum_vec1 = simd_ops::fmadd(a1, b1, sum_vec1);

//                                     simd_f32 a2 = simd_ops::load(&A[ii * k + pp + 2 * simd_ops::SIMD_WIDTH]);
//                                     simd_f32 b2 = simd_ops::load(&b[jj * k + pp + 2 * simd_ops::SIMD_WIDTH]);
//                                     sum_vec2 = simd_ops::fmadd(a2, b2, sum_vec2);

//                                     simd_f32 a3 = simd_ops::load(&A[ii * k + pp + 3 * simd_ops::SIMD_WIDTH]);
//                                     simd_f32 b3 = simd_ops::load(&b[jj * k + pp + 3 * simd_ops::SIMD_WIDTH]);
//                                     sum_vec3 = simd_ops::fmadd(a3, b3, sum_vec3);
//                                 }

//                                 for (; pp < p_end; ++pp) {
//                                     sum += A[ii * k + pp] * b[jj * k + pp];
//                                 }

//                                 sum += simd_ops::horizontal_sum(sum_vec0);
//                                 sum += simd_ops::horizontal_sum(sum_vec1);
//                                 sum += simd_ops::horizontal_sum(sum_vec2);
//                                 sum += simd_ops::horizontal_sum(sum_vec3);
//                                 // Each task writes only to its own rows, so this is safe without locks
//                                 res[ii * c + jj] += sum;
//                             }
//                         }
//                     }
//                 }
//             }
//         });

//         futures.emplace_back(std::move(fut));
//     }

//     // wait for all futures to complete
//     for (auto &f : futures) {
//         if (f.valid()) f.get();
//     }

//     // ensure pool tasks drained (not strictly necessary since futures completed)
//     pool->wait_for_completion();

//     free(b);
//     return res;
// }

float* async_matrix_mul_trans_block_with_simd(const float* A, const float* B, float* res, int r, int k, int c, int block_size = 64, int num_threads = -1) {
    const size_t align = 64;
    // Transpose B into b as other functions do
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
        for (int j = 0; j < k; ++j) {
            b[i * k + j] = B[j * c + i];
        }
    }

    // Determine number of worker threads to use
    if (num_threads <= 0) {
        unsigned int hw = std::thread::hardware_concurrency();
        num_threads = hw == 0 ? 4 : static_cast<int>(hw);
    }

    // Split work by rows: each task handles a contiguous range of rows [row_start, row_end)
    int rows_per_task = (r + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        int row_start = t * rows_per_task;
        if (row_start >= r) break;
        int row_end = std::min(r, row_start + rows_per_task);

        // capture by value the pointers and ranges to avoid race on locals
        auto fut = ThreadPool::get_instance(num_threads).enqueue_task([=, &A, &b, &res]() {
            // For the assigned row range, perform blocked, transposed, simd-accelerated multiply
            for (int i = row_start; i < row_end; i += block_size) {
                for (int j = 0; j < c; j += block_size) {
                    for (int p = 0; p < k; p += block_size) {
                        int i_end = std::min(i + block_size, row_end);
                        int j_end = std::min(j + block_size, c);
                        int p_end = std::min(p + block_size, k);

                        for (int ii = i; ii < i_end; ++ii) {
                            for (int jj = j; jj < j_end; ++jj) {
                                float sum = 0.0f;
                                int pp = p;

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

                                for (; pp < p_end; ++pp) {
                                    sum += A[ii * k + pp] * b[jj * k + pp];
                                }

                                sum += simd_ops::horizontal_sum(sum_vec0);
                                sum += simd_ops::horizontal_sum(sum_vec1);
                                sum += simd_ops::horizontal_sum(sum_vec2);
                                sum += simd_ops::horizontal_sum(sum_vec3);
                                // Each task writes only to its own rows, so this is safe without locks
                                res[ii * c + jj] += sum;
                            }
                        }
                    }
                }
            }
        });

        futures.emplace_back(std::move(fut));
    }

    // wait for all futures to complete
    for (auto &f : futures) {
        if (f.valid()) f.get();
    }

    // ensure pool tasks drained (not strictly necessary since futures completed)
    ThreadPool::get_instance().wait_for_completion();

    free(b);
    return res;
}

#ifdef __AVX__

#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <vector>
#include <immintrin.h>
#include <omp.h>

// 基于您的测试结果，最优分块大小为144x144
constexpr int OPTIMAL_BLOCK_SIZE = 144;
constexpr int AVX_REGISTER_COUNT = 16; // 16个YMM寄存器
constexpr int AVX_FLOATS_PER_REG = 8;  // 每个YMM寄存器存8个float

// 专用的AVX微内核 - 针对144x144分块优化
void avx_micro_kernel_6x16(const float* A, const float* B, float* C, 
                          int k, int ldc, int prefetch_offset = 64) {
    // 使用12个YMM寄存器作为累加器（充分利用16个可用寄存器）
    __m256 acc00, acc01, acc10, acc11, acc20, acc21;
    __m256 acc30, acc31, acc40, acc41, acc50, acc51;
    
    // 初始化12个累加器
    acc00 = _mm256_setzero_ps();
    acc01 = _mm256_setzero_ps();
    acc10 = _mm256_setzero_ps();
    acc11 = _mm256_setzero_ps();
    acc20 = _mm256_setzero_ps();
    acc21 = _mm256_setzero_ps();
    acc30 = _mm256_setzero_ps();
    acc31 = _mm256_setzero_ps();
    acc40 = _mm256_setzero_ps();
    acc41 = _mm256_setzero_ps();
    acc50 = _mm256_setzero_ps();
    acc51 = _mm256_setzero_ps();
    
    // 预取下一轮的数据
    _mm_prefetch((char*)(A + 64), _MM_HINT_T0);
    _mm_prefetch((char*)(B + 64), _MM_HINT_T0);
    
    // 主循环 - 处理K维度
    for (int p = 0; p < k; p += 4) {
        // 预取未来数据
        if (p + prefetch_offset < k) {
            _mm_prefetch((char*)(A + p + prefetch_offset), _MM_HINT_T0);
            _mm_prefetch((char*)(B + p + prefetch_offset), _MM_HINT_T0);
        }
        
        // 加载A矩阵的6行数据（广播加载）
        __m256 a0 = _mm256_broadcast_ss(A + 0 * k + p);
        __m256 a1 = _mm256_broadcast_ss(A + 1 * k + p);
        __m256 a2 = _mm256_broadcast_ss(A + 2 * k + p);
        __m256 a3 = _mm256_broadcast_ss(A + 3 * k + p);
        __m256 a4 = _mm256_broadcast_ss(A + 4 * k + p);
        __m256 a5 = _mm256_broadcast_ss(A + 5 * k + p);
        
        // 加载B矩阵的16列数据（向量加载）
        __m256 b0 = _mm256_load_ps(B + p * 16 + 0);
        __m256 b1 = _mm256_load_ps(B + p * 16 + 8);
        
        // 乘加运算 - 充分利用所有累加器
        acc00 = _mm256_add_ps(acc00, _mm256_mul_ps(a0, b0));
        acc01 = _mm256_add_ps(acc01, _mm256_mul_ps(a0, b1));
        
        acc10 = _mm256_add_ps(acc10, _mm256_mul_ps(a1, b0));
        acc11 = _mm256_add_ps(acc11, _mm256_mul_ps(a1, b1));
        
        acc20 = _mm256_add_ps(acc20, _mm256_mul_ps(a2, b0));
        acc21 = _mm256_add_ps(acc21, _mm256_mul_ps(a2, b1));
        
        acc30 = _mm256_add_ps(acc30, _mm256_mul_ps(a3, b0));
        acc31 = _mm256_add_ps(acc31, _mm256_mul_ps(a3, b1));
        
        acc40 = _mm256_add_ps(acc40, _mm256_mul_ps(a4, b0));
        acc41 = _mm256_add_ps(acc41, _mm256_mul_ps(a4, b1));
        
        acc50 = _mm256_add_ps(acc50, _mm256_mul_ps(a5, b0));
        acc51 = _mm256_add_ps(acc51, _mm256_mul_ps(a5, b1));
    }
    
    // 水平求和并存储结果
    // 对每个2x256位累加器进行水平求和，得到8个float
    __m256 sum0 = _mm256_hadd_ps(acc00, acc01);
    __m256 sum1 = _mm256_hadd_ps(acc10, acc11);
    __m256 sum2 = _mm256_hadd_ps(acc20, acc21);
    __m256 sum3 = _mm256_hadd_ps(acc30, acc31);
    __m256 sum4 = _mm256_hadd_ps(acc40, acc41);
    __m256 sum5 = _mm256_hadd_ps(acc50, acc51);
    
    // 最终存储
    _mm256_store_ps(C + 0 * ldc + 0, sum0);
    _mm256_store_ps(C + 1 * ldc + 0, sum1);
    _mm256_store_ps(C + 2 * ldc + 0, sum2);
    _mm256_store_ps(C + 3 * ldc + 0, sum3);
    _mm256_store_ps(C + 4 * ldc + 0, sum4);
    _mm256_store_ps(C + 5 * ldc + 0, sum5);
}

// 针对缓存冲突优化的分块转置
void cache_optimized_transpose(float* dst, const float* src, int rows, int cols) {
    const int cache_line_size = 64; // 64字节缓存行
    const int floats_per_line = cache_line_size / sizeof(float);
    
    // 使用缓存行对齐的转置，避免缓存冲突
    for (int i = 0; i < rows; i += floats_per_line) {
        for (int j = 0; j < cols; j += floats_per_line) {
            int i_end = std::min(i + floats_per_line, rows);
            int j_end = std::min(j + floats_per_line, cols);
            
            for (int ii = i; ii < i_end; ++ii) {
                for (int jj = j; jj < j_end; ++jj) {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }
}

// 主优化函数 - 结合OpenMP和AVX
float* optimized_matrix_mul_xeon_e5(const float* A, const float* B, 
                                   float* C, int r, int k, int c, int bs = 0) {
    if (!C) {
        C = static_cast<float*>(aligned_alloc(64, sizeof(float) * r * c));
    }
    if (!C) return nullptr;
    
    memset(C, 0, sizeof(float) * r * c);
    
    // 转置B矩阵以获得更好的内存访问模式
    float* B_transposed = static_cast<float*>(aligned_alloc(64, sizeof(float) * k * c));
    cache_optimized_transpose(B_transposed, B, k, c);
    
    // 设置OpenMP线程数（8核心）
    // 如果启用了 OpenMP，则设置线程数；否则用单线程（pragma 会被忽略）
#ifdef _OPENMP
    omp_set_num_threads(8);
#endif

    // 使用保守但正确的块乘实现（B 已转置为 c x k 布局）
    // 这段实现优先保证正确性：C[i*c + j] += A[i*k + p] * B_transposed[j*k + p]
#pragma omp parallel for schedule(dynamic)
    for (int ib = 0; ib < r; ib += OPTIMAL_BLOCK_SIZE) {
        for (int jb = 0; jb < c; jb += OPTIMAL_BLOCK_SIZE) {
            int i_end = std::min(ib + OPTIMAL_BLOCK_SIZE, r);
            int j_end = std::min(jb + OPTIMAL_BLOCK_SIZE, c);

            for (int i0 = ib; i0 < i_end; ++i0) {
                for (int j0 = jb; j0 < j_end; ++j0) {
                    float sum = 0.0f;
                    const float* a_row = A + i0 * k;
                    const float* b_row = B_transposed + j0 * k; // B_transposed is c x k
                    // 累加整个 K 维度
                    for (int p = 0; p < k; ++p) {
                        sum += a_row[p] * b_row[p];
                    }
                    C[i0 * c + j0] += sum;
                }
            }
        }
    }
    
    free(B_transposed);
    return C;
}

float* best_matrix_mul(const float* A, const float* B, float* C, int r, int k, int c, int bs = 0) {
    return optimized_matrix_mul_xeon_e5(A, B, C, r, k, c, bs);
}

#elif defined(SIMD_ARCH_APPLE_METAL)  // Apple Silicon (M1/M2) - use Accelerate

#include <Accelerate/Accelerate.h>

// Apple M2 Pro 专用矩阵乘法：调用 Accelerate 中的 cblas_sgemm，利用系统 BLAS（针对 Apple Silicon 优化）
float* optimized_matrix_mul_apple_m2(const float* A, const float* B, float* C, int r, int k, int c, int bs = 0) {
    const size_t align = 64;
    if (!C) {
        C = static_cast<float*>(aligned_alloc(align, sizeof(float) * r * c));
    }
    if (!C) return nullptr;

    // 使用 Accelerate 的 BLAS 接口执行高性能矩阵乘法：C = A * B
    // 参数说明：RowMajor, A: r x k, B: k x c, C: r x c
    // 这里我们先将 C 置零（cblas_sgemm 的 beta 参数为 0 会覆盖 C，但为保险起见先清零）
    memset(C, 0, sizeof(float) * r * c);

    // 调用 cblas_sgemm：使用单精度通用矩阵乘法，Accelerate 会在 Apple Silicon 上使用向量化与多线程
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                r, c, k,
                1.0f,
                A, k,
                B, c,
                0.0f,
                C, c);
#pragma clang diagnostic pop

    return C;
}

// 缓存优化的转置函数（需要添加到您的代码中）
void cache_optimized_transpose(float* dst, const float* src, int rows, int cols) {
    const int cache_line_size = 64;
    const int floats_per_line = cache_line_size ;
    
    // 使用缓存行对齐的分块转置
    for (int i = 0; i < rows; i += floats_per_line) {
        for (int j = 0; j < cols; j += floats_per_line) {
            int i_end = std::min(i + floats_per_line, rows);
            int j_end = std::min(j + floats_per_line, cols);
            
            for (int ii = i; ii < i_end; ++ii) {
                for (int jj = j; jj < j_end; ++jj) {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }
}

// 针对M2 Pro优化的NEON水平求和函数
inline float horizontal_sum_neon(float32x4_t v) {
#if defined(__aarch64__)
    // AArch64有专门的水平求和指令
    return vaddvq_f32(v);
#else
    // ARMv7的回退实现
    float32x2_t sum = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
#endif
}

float* my_optimized_matrix_mul(const float* A, const float* B, float* res, int r, int k, int c, int block_size = 144) {
    const size_t align = 64;
    // Transpose B into b as other functions do
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
    
    // 优化的转置策略，减少缓存冲突
    cache_optimized_transpose(b, B, k, c);

    // Determine number of worker threads to use (M2 Pro通常10-12核心)
    unsigned int hw = std::thread::hardware_concurrency();
    int num_threads = hw == 0 ? 8 : static_cast<int>(hw); // M2 Pro建议8-12线程

    // Split work by rows: each task handles a contiguous range of rows [row_start, row_end)
    int rows_per_task = (r + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        int row_start = t * rows_per_task;
        if (row_start >= r) break;
        int row_end = std::min(r, row_start + rows_per_task);

        // capture by value the pointers and ranges to avoid race on locals
        auto fut = ThreadPool::get_instance().enqueue_task([=, &A, &b, &res]() {
            // For the assigned row range, perform blocked, transposed, simd-accelerated multiply
            for (int i = row_start; i < row_end; i += block_size) {
                for (int j = 0; j < c; j += block_size) {
                    for (int p = 0; p < k; p += block_size) {
                        int i_end = std::min(i + block_size, row_end);
                        int j_end = std::min(j + block_size, c);
                        int p_end = std::min(p + block_size, k);

                        for (int ii = i; ii < i_end; ++ii) {
                            for (int jj = j; jj < j_end; ++jj) {
                                float sum = 0.0f;
                                int pp = p;

                                // 针对M2 Pro的激进NEON优化 - 使用更多累加器
                                float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
                                float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
                                float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
                                float32x4_t sum_vec3 = vdupq_n_f32(0.0f);
                                float32x4_t sum_vec4 = vdupq_n_f32(0.0f);
                                float32x4_t sum_vec5 = vdupq_n_f32(0.0f);
                                float32x4_t sum_vec6 = vdupq_n_f32(0.0f);
                                float32x4_t sum_vec7 = vdupq_n_f32(0.0f);

                                // 主循环：每次处理32个元素（8个NEON向量）
                                for (pp = p; pp + 31 < p_end; pp += 32) {
                                    // 预取数据到缓存
                                    __builtin_prefetch(&A[ii * k + pp + 64], 0, 3);
                                    __builtin_prefetch(&b[jj * k + pp + 64], 0, 3);

                                    // 加载8组A和B的数据
                                    float32x4_t a0 = vld1q_f32(&A[ii * k + pp]);
                                    float32x4_t b0 = vld1q_f32(&b[jj * k + pp]);
                                    sum_vec0 = vmlaq_f32(sum_vec0, a0, b0);

                                    float32x4_t a1 = vld1q_f32(&A[ii * k + pp + 4]);
                                    float32x4_t b1 = vld1q_f32(&b[jj * k + pp + 4]);
                                    sum_vec1 = vmlaq_f32(sum_vec1, a1, b1);

                                    float32x4_t a2 = vld1q_f32(&A[ii * k + pp + 8]);
                                    float32x4_t b2 = vld1q_f32(&b[jj * k + pp + 8]);
                                    sum_vec2 = vmlaq_f32(sum_vec2, a2, b2);

                                    float32x4_t a3 = vld1q_f32(&A[ii * k + pp + 12]);
                                    float32x4_t b3 = vld1q_f32(&b[jj * k + pp + 12]);
                                    sum_vec3 = vmlaq_f32(sum_vec3, a3, b3);

                                    float32x4_t a4 = vld1q_f32(&A[ii * k + pp + 16]);
                                    float32x4_t b4 = vld1q_f32(&b[jj * k + pp + 16]);
                                    sum_vec4 = vmlaq_f32(sum_vec4, a4, b4);

                                    float32x4_t a5 = vld1q_f32(&A[ii * k + pp + 20]);
                                    float32x4_t b5 = vld1q_f32(&b[jj * k + pp + 20]);
                                    sum_vec5 = vmlaq_f32(sum_vec5, a5, b5);

                                    float32x4_t a6 = vld1q_f32(&A[ii * k + pp + 24]);
                                    float32x4_t b6 = vld1q_f32(&b[jj * k + pp + 24]);
                                    sum_vec6 = vmlaq_f32(sum_vec6, a6, b6);

                                    float32x4_t a7 = vld1q_f32(&A[ii * k + pp + 28]);
                                    float32x4_t b7 = vld1q_f32(&b[jj * k + pp + 28]);
                                    sum_vec7 = vmlaq_f32(sum_vec7, a7, b7);
                                }

                                // 处理剩余的元素（16个一组）
                                for (; pp + 15 < p_end; pp += 16) {
                                    float32x4_t a0 = vld1q_f32(&A[ii * k + pp]);
                                    float32x4_t b0 = vld1q_f32(&b[jj * k + pp]);
                                    sum_vec0 = vmlaq_f32(sum_vec0, a0, b0);

                                    float32x4_t a1 = vld1q_f32(&A[ii * k + pp + 4]);
                                    float32x4_t b1 = vld1q_f32(&b[jj * k + pp + 4]);
                                    sum_vec1 = vmlaq_f32(sum_vec1, a1, b1);

                                    float32x4_t a2 = vld1q_f32(&A[ii * k + pp + 8]);
                                    float32x4_t b2 = vld1q_f32(&b[jj * k + pp + 8]);
                                    sum_vec2 = vmlaq_f32(sum_vec2, a2, b2);

                                    float32x4_t a3 = vld1q_f32(&A[ii * k + pp + 12]);
                                    float32x4_t b3 = vld1q_f32(&b[jj * k + pp + 12]);
                                    sum_vec3 = vmlaq_f32(sum_vec3, a3, b3);
                                }

                                // 处理剩余的元素（8个一组）
                                for (; pp + 7 < p_end; pp += 8) {
                                    float32x4_t a0 = vld1q_f32(&A[ii * k + pp]);
                                    float32x4_t b0 = vld1q_f32(&b[jj * k + pp]);
                                    sum_vec0 = vmlaq_f32(sum_vec0, a0, b0);

                                    float32x4_t a1 = vld1q_f32(&A[ii * k + pp + 4]);
                                    float32x4_t b1 = vld1q_f32(&b[jj * k + pp + 4]);
                                    sum_vec1 = vmlaq_f32(sum_vec1, a1, b1);
                                }

                                // 处理剩余的元素（4个一组）
                                for (; pp + 3 < p_end; pp += 4) {
                                    float32x4_t a0 = vld1q_f32(&A[ii * k + pp]);
                                    float32x4_t b0 = vld1q_f32(&b[jj * k + pp]);
                                    sum_vec0 = vmlaq_f32(sum_vec0, a0, b0);
                                }

                                // 处理最后的标量元素
                                for (; pp < p_end; ++pp) {
                                    sum += A[ii * k + pp] * b[jj * k + pp];
                                }

                                // 水平求和所有NEON向量
                                sum += horizontal_sum_neon(sum_vec0);
                                sum += horizontal_sum_neon(sum_vec1);
                                sum += horizontal_sum_neon(sum_vec2);
                                sum += horizontal_sum_neon(sum_vec3);
                                sum += horizontal_sum_neon(sum_vec4);
                                sum += horizontal_sum_neon(sum_vec5);
                                sum += horizontal_sum_neon(sum_vec6);
                                sum += horizontal_sum_neon(sum_vec7);
                                
                                // Each task writes only to its own rows, so this is safe without locks
                                res[ii * c + jj] += sum;
                            }
                        }
                    }
                }
            }
        });

        futures.emplace_back(std::move(fut));
    }

    // wait for all futures to complete
    for (auto &f : futures) {
        if (f.valid()) f.get();
    }

    // ensure pool tasks drained (not strictly necessary since futures completed)
    ThreadPool::get_instance().wait_for_completion();

    free(b);
    return res;
}

/*
{
    // 优化的A块广播加载
inline void load_a_block_broadcast(float32x4_t a_vec[], const float* A, 
                                 int ii, int p, int ii_end, int k) {
    for (int i = ii, reg_i = 0; i < ii_end; ++i, ++reg_i) {
        float scalar = A[i * k + p];
        a_vec[reg_i] = vdupq_n_f32(scalar);
    }
}

// 优化的B块向量加载
inline void load_b_block(float32x4_t b_vec[], const float* B_transposed,
                       int jj, int p, int jj_end, int k) {
    for (int j = jj, reg_j = 0; j < jj_end; j += 4, ++reg_j) {
        if (j + 4 <= jj_end) {
            b_vec[reg_j] = vld1q_f32(&B_transposed[j * k + p]);
        } else {
            // 边界处理
            float temp[4] = {0};
            int remaining = jj_end - j;
            for (int kk = 0; kk < remaining; ++kk) {
                temp[kk] = B_transposed[j * k + p + kk];
            }
            b_vec[reg_j] = vld1q_f32(temp);
        }
    }
}

// 优化的C块加载
inline void load_c_block(float32x4_t c_regs[][2], const float* C, 
                        int ii, int jj, int ii_end, int jj_end, int ldc) {
    for (int i = ii, reg_i = 0; i < ii_end; ++i, ++reg_i) {
        for (int j = jj, reg_j = 0; j < jj_end; j += 4, ++reg_j) {
            if (j + 4 <= jj_end) {
                // 完整向量加载
                c_regs[reg_i][reg_j] = vld1q_f32(&C[i * ldc + j]);
            } else {
                // 部分加载处理边界
                float temp[4] = {0};
                int remaining = jj_end - j;
                for (int k = 0; k < remaining; ++k) {
                    temp[k] = C[i * ldc + j + k];
                }
                c_regs[reg_i][reg_j] = vld1q_f32(temp);
            }
        }
    }
}

// 优化的C块存储
inline void store_c_block(const float32x4_t c_regs[][2], float* C,
                        int ii, int jj, int ii_end, int jj_end, int ldc) {
    for (int i = ii, reg_i = 0; i < ii_end; ++i, ++reg_i) {
        for (int j = jj, reg_j = 0; j < jj_end; j += 4, ++reg_j) {
            if (j + 4 <= jj_end) {
                vst1q_f32(&C[i * ldc + j], c_regs[reg_i][reg_j]);
            } else {
                // 边界存储
                float temp[4];
                vst1q_f32(temp, c_regs[reg_i][reg_j]);
                int remaining = jj_end - j;
                for (int k = 0; k < remaining; ++k) {
                    C[i * ldc + j + k] = temp[k];
                }
            }
        }
    }
}

// 优化的NEON外积更新
inline void neon_outer_product_update(float32x4_t c_regs[][2], 
                                    const float32x4_t a_vec[], 
                                    const float32x4_t b_vec[], 
                                    int MR, int NR_quads) {
    // 展开循环，充分利用指令级并行
    #pragma unroll(2)
    for (int i = 0; i < MR; ++i) {
        #pragma unroll(2)
        for (int j = 0; j < NR_quads; ++j) {
            // 广播A的标量到4通道向量
            float32x4_t a_broadcast = vdupq_n_f32(vgetq_lane_f32(a_vec[i], 0));
            // 乘加运算
            c_regs[i][j] = vmlaq_f32(c_regs[i][j], a_broadcast, b_vec[j]);
        }
    }
}

// 激进的NEON微内核 - 针对M2 Pro优化
void aggressive_neon_kernel(const float* A, const float* B_transposed, float* C,
                          int row_start, int row_end, int col_start, int col_end,
                          int k_start, int k_end, int ldc, int k, int block_size) {
    
    // 使用寄存器分块策略：6x8 微内核（充分利用NEON寄存器）
    constexpr int MR = 6;  // 寄存器行块大小
    constexpr int NR = 8;  // 寄存器列块大小
    
    for (int i = row_start; i < row_end; i += block_size) {
        int i_end = std::min(i + block_size, row_end);
        
        for (int j = col_start; j < col_end; j += block_size) {
            int j_end = std::min(j + block_size, col_end);
            
            // 主计算循环 - 使用寄存器分块
            for (int ii = i; ii < i_end; ii += MR) {
                int ii_end = std::min(ii + MR, i_end);
                
                for (int jj = j; jj < j_end; jj += NR) {
                    int jj_end = std::min(jj + NR, j_end);
                    
                    // 加载C矩阵的当前块到寄存器
                    float32x4_t c_regs[MR][NR/4] = {0};
                    load_c_block(c_regs, C, ii, jj, ii_end, jj_end, ldc);
                    
                    // 核心计算：K维度循环
                    for (int p = k_start; p < k_end; ++p) {
                        // 加载A的MR个元素（广播加载）
                        float32x4_t a_vec[MR];
                        load_a_block_broadcast(a_vec, A, ii, p, ii_end, k);
                        
                        // 加载B的NR个元素（向量加载）
                        float32x4_t b_vec[NR/4];
                        load_b_block(b_vec, B_transposed, jj, p, jj_end, k);
                        
                        // 外积计算：a_vec * b_vec^T，累加到c_regs
                        neon_outer_product_update(c_regs, a_vec, b_vec, MR, NR/4);
                    }
                    
                    // 存储结果回C矩阵
                    store_c_block(c_regs, C, ii, jj, ii_end, jj_end, ldc);
                }
            }
        }
    }
}

// 针对M2 Pro芯片的激进SIMD优化矩阵乘法
float* my_optimized_matrix_mul(const float* A, const float* B, float* C, int r, int k, int c, int block_size = 144) {
    const size_t align = 64;
    if (!C) {
        C = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    }
    if (!C) return nullptr;

    // 使用缓存友好的转置策略
    float* B_transposed = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * c * k));
    if (!B_transposed) return nullptr;
    
    // 优化的转置函数，减少缓存冲突
    cache_optimized_transpose(B_transposed, B, k, c);
    
    // 初始化结果矩阵
    memset(C, 0, sizeof(float) * r * c);

    // 获取硬件并发数（M2 Pro通常为10-12核心）
    unsigned int hw_cores = std::thread::hardware_concurrency();
    int num_threads = hw_cores == 0 ? 8 : static_cast<int>(hw_cores);
    
    // 基于M2 Pro的L1/L2缓存大小调整分块策略
    size_t l1_cache = get_L1d_cache_size();  // M2 Pro: 192KB per core
    size_t l2_cache = get_L2_cache_size();   // M2 Pro: 16-32MB shared
    
    // 自动调整分块大小以适应缓存
    if (block_size <= 0) {
        int optimal_block = static_cast<int>(std::sqrt(l1_cache / (3.0f * sizeof(float))));
        block_size = std::min(256, std::max(64, optimal_block));
    }

    ThreadPool& pool = ThreadPool::get_instance();
    int rows_per_thread = (r + num_threads - 1) / num_threads;
    
    std::vector<std::future<void>> futures;
    futures.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        int row_start = t * rows_per_thread;
        if (row_start >= r) break;
        int row_end = std::min(r, row_start + rows_per_thread);
        
        auto fut = pool.enqueue_task([=, &A, &B_transposed, &C]() {
            // 每个线程处理自己的行范围
            aggressive_neon_kernel(A, B_transposed, C, row_start, row_end, 
                                 0, c, 0, k, c, k, block_size);
        });
        futures.emplace_back(std::move(fut));
    }

    for (auto& f : futures) {
        if (f.valid()) f.get();
    }

    free(B_transposed);
    return C;
}
}
    */

float* best_matrix_mul(const float* A, const float* B, float* C, int r, int k, int c, int bs = 0) {
    // Prefer our custom M2 Pro optimized path when on Apple Metal platform
    return optimized_matrix_mul_apple_m2(A, B, C, r, k, c);
}

#endif // End of SIMD_ARCH_APPLE_METAL

#if defined(SIMD_ARCH_ARM_NEON)  // ARM NEON
#include <openmp_wrappers/__clang_openmp_device_functions.h>

float* optimized_matrix_mul_arm_neon(const float* A, const float* B, float* C, int r, int k, int c) {
    if (!C) {
        C = static_cast<float*>(aligned_alloc(64, sizeof(float) * r * c));
    }
    if (!C) return nullptr;

    memset(C, 0, sizeof(float) * r * c);

    // 使用 ARM NEON 指令集进行优化
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < k; p += 4) {
                // 加载 A 和 B 的数据
                float32x4_t a = vld1q_f32(A + i * k + p);
                float32x4_t b = vld1q_f32(B + j * k + p);
                // 进行乘法并累加
                sum += vaddvq_f32(vmulq_f32(a, b));
            }
            C[i * c + j] = sum;
        }
    }

    return C;
}

// 在 ARM NEON / Apple 平台上补充一个小型 NEON 微内核与 OpenMP 并行封装
#if defined(SIMD_ARCH_APPLE_METAL) || defined(SIMD_ARCH_ARM_NEON)
#include <omp.h>

// 优化的NEON微内核：避免临时数组拷贝，直接向量加载
static inline void neon_micro_kernel_4x8(const float* A_block, const float* B_block, float* C_block, 
                                         int k, int ldc, int prefetch_offset = 64) {
    // 使用8个NEON寄存器作为累加器（4行x8列）
    float32x4_t acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;
    
    // 初始化累加器
    acc00 = acc01 = vdupq_n_f32(0.0f);
    acc10 = acc11 = vdupq_n_f32(0.0f);
    acc20 = acc21 = vdupq_n_f32(0.0f);
    acc30 = acc31 = vdupq_n_f32(0.0f);
    
    // 主循环 - 处理K维度
    for (int p = 0; p < k; ++p) {
        // 预取未来数据
        if (p + prefetch_offset < k) {
            __builtin_prefetch(A_block + p + prefetch_offset, 0, 3);
            __builtin_prefetch(B_block + p + prefetch_offset, 0, 3);
        }
        
        // 加载A矩阵的4行数据（广播加载）- 优化：直接从内存加载
        float a0 = A_block[0 * k + p];
        float a1 = A_block[1 * k + p];
        float a2 = A_block[2 * k + p];
        float a3 = A_block[3 * k + p];
        float32x4_t a0v = vdupq_n_f32(a0);
        float32x4_t a1v = vdupq_n_f32(a1);
        float32x4_t a2v = vdupq_n_f32(a2);
        float32x4_t a3v = vdupq_n_f32(a3);
        
        // 从转置后的B中加载8个元素（B_block布局是8 x k）
        float btmp[8];
        for (int j = 0; j < 8; ++j) {
            btmp[j] = B_block[j * k + p];
        }
        float32x4_t b0 = vld1q_f32(btmp + 0);
        float32x4_t b1 = vld1q_f32(btmp + 4);
        
        // 乘加运算 - 充分利用所有累加器
        acc00 = vmlaq_f32(acc00, a0v, b0);
        acc01 = vmlaq_f32(acc01, a0v, b1);
        
        acc10 = vmlaq_f32(acc10, a1v, b0);
        acc11 = vmlaq_f32(acc11, a1v, b1);
        
        acc20 = vmlaq_f32(acc20, a2v, b0);
        acc21 = vmlaq_f32(acc21, a2v, b1);
        
        acc30 = vmlaq_f32(acc30, a3v, b0);
        acc31 = vmlaq_f32(acc31, a3v, b1);
    }
    
    // 存储结果到C_block
    vst1q_f32(&C_block[0 * ldc + 0], acc00);
    vst1q_f32(&C_block[0 * ldc + 4], acc01);
    vst1q_f32(&C_block[1 * ldc + 0], acc10);
    vst1q_f32(&C_block[1 * ldc + 4], acc11);
    vst1q_f32(&C_block[2 * ldc + 0], acc20);
    vst1q_f32(&C_block[2 * ldc + 4], acc21);
    vst1q_f32(&C_block[3 * ldc + 0], acc30);
    vst1q_f32(&C_block[3 * ldc + 4], acc31);
}

// NEON 4x4 微内核：每次计算 A_block (4 x k) * B_block^T (4 x k)^T -> C_block (4 x 4)
// B_block 已经转置，布局为 (4 x k)
static inline void neon_micro_kernel_4x4(const float* A_block, const float* B_block, float* C_block, int k, int ldc) {
    float32x4_t c0 = vdupq_n_f32(0.0f);
    float32x4_t c1 = vdupq_n_f32(0.0f);
    float32x4_t c2 = vdupq_n_f32(0.0f);
    float32x4_t c3 = vdupq_n_f32(0.0f);

    for (int p = 0; p < k; ++p) {
        // 从转置后的 B 中加载当前 p 对应的 4 个元素
        // B_block 布局是 4 x k，所以 B_block[j * k + p] 访问第 j 行第 p 列
        float btmp[4];
        btmp[0] = B_block[0 * k + p];
        btmp[1] = B_block[1 * k + p];
        btmp[2] = B_block[2 * k + p];
        btmp[3] = B_block[3 * k + p];
        float32x4_t bvec = vld1q_f32(btmp);

        // 广播 A 的 4 个标量到向量
        float a0 = A_block[0 * k + p];
        float a1 = A_block[1 * k + p];
        float a2 = A_block[2 * k + p];
        float a3 = A_block[3 * k + p];
        float32x4_t a0v = vdupq_n_f32(a0);
        float32x4_t a1v = vdupq_n_f32(a1);
        float32x4_t a2v = vdupq_n_f32(a2);
        float32x4_t a3v = vdupq_n_f32(a3);

        // 乘加运算：C[i][j] += A[i][p] * B[j][p]
        c0 = vmlaq_f32(c0, a0v, bvec);
        c1 = vmlaq_f32(c1, a1v, bvec);
        c2 = vmlaq_f32(c2, a2v, bvec);
        c3 = vmlaq_f32(c3, a3v, bvec);
    }

    // 存储结果到 C_block
    vst1q_f32(&C_block[0 * ldc], c0);
    vst1q_f32(&C_block[1 * ldc], c1);
    vst1q_f32(&C_block[2 * ldc], c2);
    vst1q_f32(&C_block[3 * ldc], c3);
}

// OpenMP + 分块的 NEON 矩阵乘法
float* optimized_matrix_mul_neon_omp(const float* A, const float* B, float* C, int r, int k, int c, int block_size = 128) {
    const size_t align = 64;
    
    if (!C) {
        C = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    }
    if (!C) return nullptr;
    
    memset(C, 0, sizeof(float) * r * c);
    
    // 转置B矩阵以获得更好的内存访问模式
    float* B_transposed = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * k * c));
    if (!B_transposed) {
        if (C) free(C);
        return nullptr;
    }
    
    // 转置B矩阵：B[j][i] -> B_transposed[i][j]
    cache_optimized_transpose(B_transposed, B, k, c);
    
    // 设置OpenMP线程数
#ifdef _OPENMP
    omp_set_num_threads(12); // 使用较少线程避免过度并行
#endif

    // 使用NEON微内核执行块乘（B已转置为c x k布局）
    // 主循环按块遍历：对每个 block 内再按 micro-kernel 的大小步进
    // micro-kernel: 4x8 (4 行 * 8 列)，当遇到边界余数时回退到标量实现
#pragma omp parallel for collapse(2) schedule(dynamic)
    for (int ib = 0; ib < r; ib += block_size) {
        for (int jb = 0; jb < c; jb += block_size) {
            int i_end = std::min(ib + block_size, r);
            int j_end = std::min(jb + block_size, c);

            for (int i0 = ib; i0 < i_end; i0 += 4) {
                int im = std::min(4, i_end - i0);
                for (int j0 = jb; j0 < j_end; j0 += 8) {
                    int jn = std::min(8, j_end - j0);

                    if (im == 4 && jn == 8) {
                        // 指向 A 的 4 行起始（每行长度为 k）
                        const float* A_block = A + i0 * k;
                        // 指向转置后 B 的对应 8 行（每行长度为 k），B_transposed 布局为 c x k
                        const float* B_block = B_transposed + j0 * k;
                        // 指向 C 矩阵块的左上角（行主序，行长为 c）
                        float* C_block = C + i0 * c + j0;

                        // 预取 A_block 的开始与 B_block 的开始，减少缓存缺失
                        __builtin_prefetch(A_block, 0, 3);
                        __builtin_prefetch(B_block, 0, 3);

                        // 调用 NEON 微内核
                        neon_micro_kernel_4x8(A_block, B_block, C_block, k, c, 64);
                    } else {
                        // 边界余数：回退到优化的NEON向量化实现
                        for (int ii = 0; ii < im; ++ii) {
                            for (int jj = 0; jj < jn; ++jj) {
                                float sum = 0.0f;
                                const float* a_row = A + (i0 + ii) * k;
                                const float* b_row = B_transposed + (j0 + jj) * k;
                                
                                // 预取数据
                                __builtin_prefetch(a_row + 64, 0, 3);
                                __builtin_prefetch(b_row + 64, 0, 3);
                                
                                // 使用NEON向量化处理主循环
                                int p = 0;
                                float32x4_t sum_vec = vdupq_n_f32(0.0f);
                                
                                // 处理4个元素为一组的向量运算
                                for (; p <= k - 4; p += 4) {
                                    float32x4_t a_vec = vld1q_f32(a_row + p);
                                    float32x4_t b_vec = vld1q_f32(b_row + p);
                                    sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
                                }
                                
                                // 水平求和NEON向量
#if defined(__aarch64__)
                                float vec_sum = vaddvq_f32(sum_vec);
#else
                                float32x2_t lo = vget_low_f32(sum_vec);
                                float32x2_t hi = vget_high_f32(sum_vec);
                                float32x2_t t = vadd_f32(lo, hi);
                                float vec_sum = vget_lane_f32(t, 0) + vget_lane_f32(t, 1);
#endif
                                
                                sum = vec_sum;
                                
                                // 处理剩余元素
                                for (; p < k; ++p) {
                                    sum += a_row[p] * b_row[p];
                                }
                                
                                C[(i0 + ii) * c + (j0 + jj)] += sum;
                            }
                        }
                    }
                }
            }
        }
    }
    
    free(B_transposed);
    return C;
}
#endif

#else

float* best_matrix_mul(const float* A, const float* B, float* C, int r, int k, int c) {
    return async_matrix_mul_trans_block_with_simd(A, B, C, r, k, c, 64);
}

#endif



float rand_float(float s) {
    return 4 * s * (1 - s);
}

float* random_matrix(int r, int c, float seed) {
    const size_t align = 64;
    float *res = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    if (!res) return nullptr;
    for (int i = 0; i < r * c; ++i) {
        res[i] = rand_float(seed);
    }
    return res;
}

void matrix_gen(float *a,float *b,int N,float seed){
	float s=seed;
	for(int i=0;i<N*N;i++){
		s=rand_float(s);
		a[i]=s;
		s=rand_float(s);
		b[i]=s;
	}
}

float Trace(const float* A, int r, int c) {
    float sum = 0.0f;
    int n = std::min(r, c);
    for (int i = 0; i < n; ++i) {
        sum += A[i * c + i];
    }
    return sum;
}

void print_matrix(const float* A, int r, int c) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            std::cout << A[i * c + j] << " ";
        }
        std::cout << std::endl;
    }
}

bool comp(float *a, float *b, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(a[i] - b[i]) > 1e-3) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

// 显示详细用法信息
inline void print_help() {
    std::cout << "Matrix Multiplication Performance Tester\n\n";
    std::cout << "Usage:\n";
    std::cout << "  ./program [OPTIONS] [SIZES...]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -h, --help          Show this help message\n";
    std::cout << "  -t, --test          Run in test mode, with clear guidance on usage to check out the best fit block size\n";
    std::cout << "  -s, --seed          Random seed\n";
    std::cout << "  -m, --methods LIST   Comma-separated list of methods to test\n";
    std::cout << "                       Available methods: base, block, transpose, transpose_block, simd, async_simd, best\n";
    std::cout << "                       Default: all methods, random seed 0.1f\n\n";
    std::cout << "Examples:\n";
    std::cout << "  ./program 512 1024          # Test all methods with sizes 512 and 1024\n";
    std::cout << "  ./program -m base,simd 512 # Test only base and simd methods with size 512\n";
    std::cout << "  ./program --help           # Show this help\n\n";
    std::cout << "Available methods:\n";
    std::cout << "  base            : Basic triple-loop matrix multiplication\n";
    std::cout << "  block           : Blocked matrix multiplication\n";
    std::cout << "  transpose       : Matrix multiplication with transposed B\n";
    std::cout << "  transpose_block : Blocked matrix multiplication with transposed B\n";
    std::cout << "  simd            : SIMD-optimized matrix multiplication\n";
    std::cout << "  async_simd      : Asynchronous SIMD-optimized matrix multiplication\n";
    #ifdef SIMD_ARCH_APPLE_METAL
    std::cout << "  my_optimized    : Apple M2 Pro optimized method using custom NEON kernel\n";
    std::cout << "  apple_accelerate : Apple M2 Pro optimized method using Accelerate framework\n";
    #endif
    #if defined(SIMD_ARCH_ARM_NEON)
    std::cout << "  neon_omp        : OpenMP + NEON optimized matrix multiplication\n";
    #endif
    std::cout << "  best            : Best optimized method for current hardware\n";
}

// 解析逗号分隔的方法列表
std::set<std::string> parse_methods(const std::string& method_list) {
    std::set<std::string> methods;
    std::stringstream ss(method_list);
    std::string method;
    
    while (std::getline(ss, method, ',')) {
        // 去除前后空格
        method.erase(0, method.find_first_not_of(" \t"));
        method.erase(method.find_last_not_of(" \t") + 1);
        
        if (!method.empty()) {
            methods.insert(method);
        }
    }
    
    return methods;
}

// 检查方法是否有效
bool is_valid_method(const std::string& method) {
    static const std::set<std::string> valid_methods = {
        "base", "block", "transpose", "transpose_block", "simd", "async_simd", "best"
        #ifdef SIMD_ARCH_APPLE_METAL
        , "my_optimized", "apple_accelerate"
        , "neon_omp"
        #endif
    };
    return valid_methods.find(method) != valid_methods.end();
}

void test_mod(int argc, char** argv) {
    // 默认参数
    int num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    int block_size = 64;
    std::vector<int> test_sizes = {512, 1024, 2048, 4096};
    float seed = 0.3f;
    int times = 3;
    std::vector<int> thread_counts = {1, 2, 4, 8, 16};  // 测试的线程数列表
    
    // 解析命令行参数
    for (int i = 0; i < argc; ++i) {
        if ((strcmp(argv[i], "--threads") == 0 || strcmp(argv[i], "-t") == 0) && i + 1 < argc) {
            num_threads = atoi(argv[i + 1]);
            if (num_threads <= 0) {
                std::cerr << "Invalid thread count. Using default: " << std::thread::hardware_concurrency() << std::endl;
                num_threads = std::thread::hardware_concurrency();
            }
            ++i;
        }
        else if ((strcmp(argv[i], "--block-size") == 0 || strcmp(argv[i], "-b") == 0) && i + 1 < argc) {
            block_size = atoi(argv[i + 1]);
            if (block_size <= 0) {
                std::cerr << "Invalid block size. Using default: 64" << std::endl;
                block_size = 64;
            }
            ++i;
        }
        else if ((strcmp(argv[i], "--sizes") == 0 || strcmp(argv[i], "-s") == 0) && i + 1 < argc) {
            test_sizes.clear();
            std::string sizes_str = argv[i + 1];
            std::stringstream ss(sizes_str);
            std::string size_str;
            while (std::getline(ss, size_str, ',')) {
                int size = atoi(size_str.c_str());
                if (size > 0) {
                    test_sizes.push_back(size);
                }
            }
            if (test_sizes.empty()) {
                std::cerr << "Invalid sizes. Using default: 512,1024,2048,4096" << std::endl;
                test_sizes = {512, 1024, 2048, 4096};
            }
            ++i;
        }
        else if ((strcmp(argv[i], "--seed") == 0 || strcmp(argv[i], "-e") == 0) && i + 1 < argc) {
            seed = atof(argv[i + 1]);
            ++i;
        }
        else if ((strcmp(argv[i], "--times") == 0 || strcmp(argv[i], "-n") == 0) && i + 1 < argc) {
            times = atoi(argv[i + 1]);
            if (times <= 0) {
                std::cerr << "Invalid times value. Using default: 3" << std::endl;
                times = 3;
            }
            ++i;
        }
        else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -t, --threads <num>     Set number of threads (default: hardware concurrency)" << std::endl;
            std::cout << "  -b, --block-size <size> Set block size (default: 64)" << std::endl;
            std::cout << "  -s, --sizes <list>      Set matrix sizes (comma-separated, default: 512,1024,2048,4096)" << std::endl;
            std::cout << "  -e, --seed <value>      Set random seed (default: 0.3)" << std::endl;
            std::cout << "  -n, --times <num>       Set number of runs per test (default: 3)" << std::endl;
            std::cout << "  -h, --help              Show this help message" << std::endl;
            return;
        }
    }
    
    // 输出测试配置
    std::cout << "=== async_simd Matrix Multiplication Performance Test ===" << std::endl;
    std::cout << "Method: async_matrix_mul_trans_block_with_simd" << std::endl;
    std::cout << "Block size: " << block_size << std::endl;
    std::cout << "Random seed: " << seed << std::endl;
    std::cout << "Runs per test: " << times << std::endl;
    std::cout << "Thread counts to test: ";
    for (int tc : thread_counts) std::cout << tc << " ";
    std::cout << std::endl;
    std::cout << "Matrix sizes: ";
    for (int size : test_sizes) std::cout << size << " ";
    std::cout << std::endl << std::endl;
    
    // 输出表头
    std::cout << "Matrix Size\tThreads\tTime (ms)\tGFlops\t\tTrace" << std::endl;
    std::cout << "-----------\t-------\t---------\t-------\t\t-----" << std::endl;
    
    // 对每个矩阵大小和线程数进行测试
    for (int size : test_sizes) {
        // 分配矩阵内存
        float *A = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * size * size));
        float *B = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * size * size));
        float *C = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * size * size));
        
        if (!A || !B || !C) {
            std::cerr << "Error: Failed to allocate memory for matrices of size " << size << std::endl;
            continue;
        }
        
        // 生成测试矩阵
        matrix_gen(A, B, size, seed);
        
        for (int threads : thread_counts) {
            // 只对4096矩阵测试所有线程数，其他大小只测试默认线程数
            if (size != 4096 && threads != num_threads) {
                continue;
            }
            
            double total_time = 0.0;
            float trace = 0.0f;
            
            // 运行多次取平均值
            for (int run = 0; run < times; ++run) {
                // 清零结果矩阵
                memset(C, 0, sizeof(float) * size * size);
                
                auto start = std::chrono::high_resolution_clock::now();
                
                // 调用async_simd方法，传递指定的线程数
                float* result = async_matrix_mul_trans_block_with_simd(A, B, C, size, size, size, block_size, threads);
                
                auto end = std::chrono::high_resolution_clock::now();
                double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
                total_time += elapsed_ms;
                
                // 计算trace用于验证正确性
                if (run == 0) {  // 只在第一次运行时计算trace
                    trace = 0.0f;
                    for (int i = 0; i < size; ++i) {
                        trace += result[i * size + i];
                    }
                }
            }
            
            double avg_time_ms = total_time / times;
            double gflops = (2.0 * size * size * size) / (avg_time_ms * 1e6);
            
            // 输出结果，格式化为表格形式
            printf("%-11d\t%-7d\t%-9.2f\t%-7.2f\t\t%.2e\n", 
                   size, threads, avg_time_ms, gflops, trace);
        }
        
        // 释放内存
        free(A);
        free(B);
        free(C);
    }
    
    std::cout << std::endl << "Test completed." << std::endl;
}

int main(int argc, char** argv) {
    // 打印当前使用的SIMD架构
#if defined(SIMD_ARCH_ARM_NEON)
    std::cout << "Using ARM NEON SIMD instructions" << std::endl;
#elif defined(SIMD_ARCH_X86_SSE)
    #ifdef __AVX__
        std::cout << "Using x86 AVX SIMD instructions" << std::endl;
    #else
        std::cout << "Using x86 SSE SIMD instructions" << std::endl;
    #endif
#elif defined(SIMD_ARCH_APPLE_METAL)
    std::cout << "Using Apple Metal SIMD instructions" << std::endl;
#else
    std::cout << "Using generic scalar implementation (no SIMD)" << std::endl;
#endif
    std::cout << std::endl;

    // 解析命令行参数
    std::vector<int> sizes;
    std::set<std::string> selected_methods;
    std::map<std::string, float* (*)(const float*, const float*, float*, int, int, int, int)> method_map = {
        {"base", base_mul},
        {"block", matrix_mul_block},
        {"transpose", matrix_mul_trans},
        {"transpose_block", matrix_mul_trans_block},
        {"simd", matrix_mul_trans_block_with_simd},
        {"async_simd", [](const float* A, const float* B, float* C, int r, int k, int c, int block_size) -> float* { return async_matrix_mul_trans_block_with_simd(A, B, C, r, k, c, block_size); }},
        {"best", best_matrix_mul}
        #ifdef SIMD_ARCH_APPLE_METAL
        , {"my_optimized", my_optimized_matrix_mul}
        , {"apple_accelerate", optimized_matrix_mul_apple_m2}
        , {"neon_omp", optimized_matrix_mul_neon_omp}
        #endif
        #if defined(SIMD_ARCH_ARM_NEON)
        , {"neon_omp", optimized_matrix_mul_neon_omp}
        #endif
    };
    bool show_help = false;
    float seed = 0.1f;
    int block_size = 128;

    if (argc >= 2 && (strcmp(argv[1], "-t") == 0 || strcmp(argv[1], "--test") == 0)) {
        test_mod(argc - 2, argv + 2);
        return 0;
    }
    
    // 默认选择所有方法
    selected_methods = {"base", "block", "transpose", "transpose_block", "simd", "async_simd", "best", "my_optimized", "apple_accelerate", "neon_omp"};
    
    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            show_help = true;
        } else if (arg == "-m" || arg == "--methods") {
            if (i + 1 < argc) {
                std::string method_list = argv[++i];
                auto methods = parse_methods(method_list);
                
                // 检查所有方法是否有效
                bool all_valid = true;
                for (const auto& method : methods) {
                    if (!is_valid_method(method)) {
                        std::cerr << "Error: Invalid method '" << method << "'\n";
                        all_valid = false;
                    }
                }
                
                if (all_valid && !methods.empty()) {
                    selected_methods = methods;
                } else if (methods.empty()) {
                    std::cerr << "Error: No valid methods specified\n";
                    print_help();
                    return 1;
                } else {
                    std::cerr << "Error: Some methods are invalid\n";
                    print_help();
                    return 1;
                }
            } else {
                std::cerr << "Error: -m option requires a method list\n";
                print_help();
                return 1;
            }
        } else if (arg == "-b" || arg == "--block-size") {
            if (i + 1 < argc) {
                block_size = std::stoi(argv[++i]);
            } else {
                std::cerr << "Error: -b option requires a block size value\n";
                print_help();
                return 1;
            }
        } else if (arg == "-s" || arg == "--seed") {
            if (i + 1 < argc) {
                seed = std::stof(argv[++i]);
            } else {
                std::cerr << "Error: -s option requires a seed value\n";
                print_help();
                return 1;
            }
        } else if (arg[0] == '-') {
            std::cerr << "Error: Unknown option '" << arg << "'\n";
            print_help();
            return 1;
        } else {
            // 尝试解析为数字
            try {
                int size = std::stoi(arg);
                if (size > 0) {
                    sizes.push_back(size);
                } else {
                    std::cerr << "Error: Invalid size '" << arg << "' (must be positive)\n";
                    print_help();
                    return 1;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error: Invalid argument '" << arg << "'\n";
                print_help();
                return 1;
            }
        }
    }
    
    // 显示帮助信息
    if (show_help) {
        print_help();
        return 0;
    }
    
    // 检查是否有有效的大小参数
    if (sizes.empty()) {
        std::cerr << "Error: No matrix sizes specified\n";
        print_help();
        return 1;
    }
    
    // 显示选定的方法
    std::cout << "Selected methods: ";
    for (const auto& method : selected_methods) {
        std::cout << method << " ";
    }
    std::cout << "\n\n";

    // 执行测试
    for (int size : sizes) {
        float *A = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * size * size));
        float *B = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * size * size));
        if (!A || !B) {
            std::cerr << "Error: Failed to allocate memory for matrices of size " << size << std::endl;
            continue;
        }
        matrix_gen(A,B,size,seed);
        std::cout << "Matrix multiplication performance comparison (" << size << " x " << size << " matrices):" << std::endl;

        for (const auto& method : selected_methods) {
            std::cout << "- " << method << std::endl;
            #define TEST
            #ifdef TEST
            timer t;
            t.start_timer();
            #endif
            auto C = method_map[method](A, B, nullptr, size, size, size, block_size);
            #ifdef TEST
            t.end_timer();
            #endif
            float trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            free(C);
            std::cout << std::endl;
        }

/*
        // 测试选定的方法
        if (selected_methods.count("base")) {
            std::cout << "Base multiplication:" << std::endl;
            timer t;
            auto C = base_mul(A, B, nullptr, size, size, size);
            t.end_timer();
            float trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            free(C);
            std::cout << std::endl;
        }

        if (selected_methods.count("block")) {
            std::cout << "Block multiplication:" << std::endl;
            timer t;
            auto C = matrix_mul_block(A, B, nullptr, size, size, size);
            t.end_timer();
            float trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            free(C);
            std::cout << std::endl;
        }

        if (selected_methods.count("transpose")) {
            std::cout << "Transpose multiplication:" << std::endl;
            timer t;
            auto C = matrix_mul_trans(A, B, nullptr, size, size, size);
            t.end_timer();
            float trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            free(C);
            std::cout << std::endl;
        }

        if (selected_methods.count("transpose_block")) {
            std::cout << "Transpose + Block multiplication:" << std::endl;
            timer t;
            auto C = matrix_mul_trans_block(A, B, nullptr, size, size, size);
            t.end_timer();
            float trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            free(C);
            std::cout << std::endl;
        }

        if (selected_methods.count("simd")) {
            std::cout << "SIMD multiplication:" << std::endl;
            timer t;
            auto C = matrix_mul_trans_block_with_simd(A, B, nullptr, size, size, size);
            t.end_timer();
            float trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            free(C);
            std::cout << std::endl;
        }

        if (selected_methods.count("async_simd")) {
            std::cout << "Asynchronous SIMD multiplication:" << std::endl;
            ThreadPool& pool = ThreadPool::get_instance();
            timer t;
            auto C = async_matrix_mul_trans_block_with_simd(A, B, nullptr, size, size, size, block_size);
            t.end_timer();
            float trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            float *bc = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * size * size));
            base_mul(bc, A, B, size, size, size);
            if (comp(C, bc, size)) {
                std::cout << "Result verification passed." << std::endl;
            } else {
                std::cout << "Result verification failed!" << std::endl;
            }
            free(bc);
            free(C);
            std::cout << std::endl;
        }

        #ifdef SIMD_ARCH_APPLE_METAL
        if (selected_methods.count("my_optimized")) {
            std::cout << "Apple M2 Pro optimized multiplication (custom NEON kernel):" << std::endl;
            timer t;
            auto C = my_optimized_matrix_mul(A, B, nullptr, size, size, size, block_size);
            t.end_timer();
            float trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            free(C);
            std::cout << std::endl;
        }

        if (selected_methods.count("apple_accelerate")) {
            std::cout << "Apple M2 Pro optimized multiplication (Accelerate framework):" << std::endl;
            timer t;
            auto C = optimized_matrix_mul_apple_m2(A, B, nullptr, size, size, size);
            t.end_timer();
            float trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            free(C);
            std::cout << std::endl;
        }
        #endif

        if (selected_methods.count("best")) {
            std::cout << "Best (hardware-optimized) multiplication:" << std::endl;
            timer t;
            auto C = best_matrix_mul(A, B, nullptr, size, size, size);
            t.end_timer();
            float trace = Trace(C, size, size);
            std::cout << "Trace: " << trace << std::endl;
            free(C);
            std::cout << std::endl;
        }
*/
        
        free(A);
        free(B);
    }
    
    return 0;
}