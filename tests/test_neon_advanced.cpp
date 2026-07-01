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

// NEON 4x8 微内核：每次计算 A_block (4 x k) * B_block^T (8 x k)^T -> C_block (4 x 8)
// 针对ARM NEON优化的高性能微内核，使用数据预取
static inline void neon_micro_kernel_4x8(const float* A_block, const float* B_block, float* C_block, 
                                         int k, int ldc, int prefetch_offset = 64) {
    // 使用8个NEON寄存器作为累加器（4行x8列）
    float32x4_t acc00, acc01, acc10, acc11, acc20, acc21, acc30, acc31;
    
    // 初始化累加器
    acc00 = vdupq_n_f32(0.0f);
    acc01 = vdupq_n_f32(0.0f);
    acc10 = vdupq_n_f32(0.0f);
    acc11 = vdupq_n_f32(0.0f);
    acc20 = vdupq_n_f32(0.0f);
    acc21 = vdupq_n_f32(0.0f);
    acc30 = vdupq_n_f32(0.0f);
    acc31 = vdupq_n_f32(0.0f);
    
    // 预取下一轮的数据
    __builtin_prefetch(A_block + 64, 0, 3);
    __builtin_prefetch(B_block + 64, 0, 3);
    
    // 主循环 - 处理K维度
    for (int p = 0; p < k; ++p) {
        // 预取未来数据
        if (p + prefetch_offset < k) {
            __builtin_prefetch(A_block + p + prefetch_offset, 0, 3);
            __builtin_prefetch(B_block + p + prefetch_offset, 0, 3);
        }
        
        // 加载A矩阵的4行数据（广播加载）
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

// 优化后的NEON + OpenMP矩阵乘法（带微内核和数据预取）
float* optimized_matrix_mul_neon_omp_advanced(const float* A, const float* B, float* C, int r, int k, int c, int block_size = 128) {
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
    omp_set_num_threads(4);
#endif

    // 使用NEON微内核执行块乘（B已转置为c x k布局）
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

#else
float* optimized_matrix_mul_neon_omp_advanced(const float* A, const float* B, float* C, int r, int k, int c, int block_size = 128) {
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
    const int N = 512;
    const size_t align = 64;
    
    std::cout << "Testing Advanced NEON + OpenMP matrix multiplication with size " << N << "x" << N << std::endl;
    
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
    float* result = optimized_matrix_mul_neon_omp_advanced(A, B, C_neon, N, N, N);
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
        std::cout << "Advanced NEON + OpenMP optimization is CORRECT!" << std::endl;
        std::cout << "Base implementation time: " << duration_base.count() << " ms" << std::endl;
        std::cout << "Advanced NEON + OpenMP time: " << duration_neon.count() << " ms" << std::endl;
        if (duration_base.count() > 0) {
            double speedup = duration_base.count() / duration_neon.count();
            std::cout << "Speedup: " << speedup << "x" << std::endl;
        }
        
        // 计算GFLOPS
        double gflops = (2.0 * N * N * N) / (duration_neon.count() / 1000.0) / 1e9;
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    } else {
        std::cout << "Advanced NEON + OpenMP optimization has ERRORS!" << std::endl;
    }
    
    // 释放内存
    free(A);
    free(B);
    free(C_neon);
    free(C_base);
    
    return correct ? 0 : 1;
}