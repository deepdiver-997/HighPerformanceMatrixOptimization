#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

static void* aligned_alloc_helper(size_t align, size_t size) {
#if defined(_ISOC11_SOURCE) || (defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L)
    size_t sz = ((size + align - 1) / align) * align;
    return aligned_alloc(align, sz);
#else
    void* ptr = nullptr;
    size_t sz = ((size + align - 1) / align) * align;
    if (posix_memalign(&ptr, align, sz) != 0) return nullptr;
    return ptr;
#endif
}

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>

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
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < k; ++j) {
            B_transposed[i * k + j] = B[j * c + i];
        }
    }
    
    // 设置OpenMP线程数
#ifdef _OPENMP
    omp_set_num_threads(4); // 减少线程数避免过度并行
#endif

    // 使用分块和NEON向量化
#pragma omp parallel for collapse(2) schedule(static)
    for (int ib = 0; ib < r; ib += block_size) {
        for (int jb = 0; jb < c; jb += block_size) {
            int i_end = std::min(ib + block_size, r);
            int j_end = std::min(jb + block_size, c);

            for (int i = ib; i < i_end; ++i) {
                for (int j = jb; j < j_end; ++j) {
                    float sum = 0.0f;
                    const float* a_row = A + i * k;
                    const float* b_row = B_transposed + j * k;
                    
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
                    
                    C[i * c + j] += sum;
                }
            }
        }
    }
    
    free(B_transposed);
    return C;
}

#else
float* optimized_matrix_mul_neon_omp(const float* A, const float* B, float* C, int r, int k, int c, int block_size = 128) {
    std::cout << "NEON not supported on this platform" << std::endl;
    return nullptr;
}
#endif

void base_mul(float* C, const float* A, const float* B, int r, int k, int c) {
    for (int i = 0; i < r; ++i) {
        for (int j = 0; j < c; ++j) {
            float tmp = 0.0f;
            for (int p = 0; p < k; ++p) {
                tmp += A[i * k + p] * B[p * c + j];
            }
            C[i * c + j] = tmp;
        }
    }
}

bool compare_matrices(const float* A, const float* B, int size, float tolerance = 1e-3f) {
    for (int i = 0; i < size * size; ++i) {
        if (std::abs(A[i] - B[i]) > tolerance) {
            std::cout << "Mismatch at index " << i << ": " << A[i] << " vs " << B[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 128;
    const size_t align = 64;
    
    std::cout << "Testing NEON + OpenMP matrix multiplication with size " << N << "x" << N << std::endl;
    
#ifdef _OPENMP
    std::cout << "OpenMP is available, using " << omp_get_max_threads() << " threads" << std::endl;
#else
    std::cout << "OpenMP is not available" << std::endl;
#endif
    
    // 分配内存
    float* A = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * N * N));
    float* B = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * N * N));
    float* C_neon = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * N * N));
    float* C_base = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * N * N));
    
    if (!A || !B || !C_neon || !C_base) {
        std::cout << "Memory allocation failed" << std::endl;
        return 1;
    }
    
    // 初始化矩阵
    for (int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(i % 100) / 100.0f;
        B[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    }
    
    // 计算基准结果
    auto start_base = std::chrono::high_resolution_clock::now();
    base_mul(C_base, A, B, N, N, N);
    auto end_base = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_base = end_base - start_base;
    
    // 计算NEON优化结果
    auto start_neon = std::chrono::high_resolution_clock::now();
    float* result = optimized_matrix_mul_neon_omp(A, B, C_neon, N, N, N);
    auto end_neon = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration_neon = end_neon - start_neon;
    
    if (!result) {
        std::cout << "NEON optimization failed" << std::endl;
        free(A);
        free(B);
        free(C_neon);
        free(C_base);
        return 1;
    }
    
    // 比较结果
    bool correct = compare_matrices(C_neon, C_base, N);
    
    if (correct) {
        std::cout << "NEON + OpenMP optimization is CORRECT!" << std::endl;
        std::cout << "Base implementation time: " << duration_base.count() << " ms" << std::endl;
        std::cout << "NEON + OpenMP time: " << duration_neon.count() << " ms" << std::endl;
        if (duration_base.count() > 0) {
            double speedup = duration_base.count() / duration_neon.count();
            std::cout << "Speedup: " << speedup << "x" << std::endl;
        }
    } else {
        std::cout << "NEON + OpenMP optimization has ERRORS!" << std::endl;
    }
    
    // 释放内存
    free(A);
    free(B);
    free(C_neon);
    free(C_base);
    
    return correct ? 0 : 1;
}