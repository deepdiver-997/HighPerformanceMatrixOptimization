#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <vector>
#include <immintrin.h>
#include <omp.h>
#include <algorithm>

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

float rand_float(float s) { return 4.0f * s * (1.0f - s); }

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
    for (int i = 0; i < n; ++i) sum += A[i * c + i];
    return sum;
}

// 全局打包 B 矩阵
// 将 B 转换为 N/16 个块，每个块包含 16 列
// 块内布局: Row-Major (K 行，每行 16 个元素)
// B_packed 大小: K * N (假设 N 是 16 的倍数，如果不是需要 padding)
void pack_B_global(float* B_packed, const float* B, int N, int K) {
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < N; j += 16) {
        int cols = std::min(16, N - j);
        // 计算当前块在 B_packed 中的起始位置
        // 第 j/16 个块，每个块大小 K * 16
        float* dst_block = B_packed + (j / 16) * K * 16;
        
        if (cols == 16) {
            for (int k = 0; k < K; ++k) {
                const float* src = B + k * N + j;
                float* dst = dst_block + k * 16;
                _mm256_storeu_ps(dst + 0, _mm256_loadu_ps(src + 0));
                _mm256_storeu_ps(dst + 8, _mm256_loadu_ps(src + 8));
            }
        } else {
            for (int k = 0; k < K; ++k) {
                const float* src = B + k * N + j;
                float* dst = dst_block + k * 16;
                for (int c = 0; c < cols; ++c) dst[c] = src[c];
                for (int c = cols; c < 16; ++c) dst[c] = 0.0f;
            }
        }
    }
}

// AVX 微内核（针对已打包的 B 矩阵）
// B_packed 布局为 (k x 16), 即 Row-Major 的 16 列条带
// C 布局为 (M x N) 的一部分
__attribute__((always_inline))
inline void avx_micro_kernel_6x16_packed(int k_size, const float* A, int lda, const float* B_packed, float* C, int ldc) {
    __m256 c[6][2];
    // 初始化累加器
    for (int i = 0; i < 6; ++i) {
        c[i][0] = _mm256_loadu_ps(C + i * ldc + 0);
        c[i][1] = _mm256_loadu_ps(C + i * ldc + 8);
    }

    int p = 0;
    // 4路循环展开
    for (; p + 3 < k_size; p += 4) {
        // 预取 B (提前约 16-32 个 float)
        _mm_prefetch((const char*)(B_packed + (p + 8) * 16), _MM_HINT_T0);
        // 预取 A
        _mm_prefetch((const char*)(A + 0 * lda + p + 16), _MM_HINT_T0);
        _mm_prefetch((const char*)(A + 3 * lda + p + 16), _MM_HINT_T0);

        // 预加载 B 的 4 行 (每行 16 个元素)
        const float* bp0 = B_packed + (p + 0) * 16;
        const float* bp1 = B_packed + (p + 1) * 16;
        const float* bp2 = B_packed + (p + 2) * 16;
        const float* bp3 = B_packed + (p + 3) * 16;

        __m256 b0_0 = _mm256_load_ps(bp0 + 0);
        __m256 b0_1 = _mm256_load_ps(bp0 + 8);
        __m256 b1_0 = _mm256_load_ps(bp1 + 0);
        __m256 b1_1 = _mm256_load_ps(bp1 + 8);
        __m256 b2_0 = _mm256_load_ps(bp2 + 0);
        __m256 b2_1 = _mm256_load_ps(bp2 + 8);
        __m256 b3_0 = _mm256_load_ps(bp3 + 0);
        __m256 b3_1 = _mm256_load_ps(bp3 + 8);

        // 对 A 的 6 行进行计算
        for (int i = 0; i < 6; ++i) {
            const float* a_ptr = A + i * lda + p;
            __m256 a0 = _mm256_broadcast_ss(a_ptr + 0);
            __m256 a1 = _mm256_broadcast_ss(a_ptr + 1);
            __m256 a2 = _mm256_broadcast_ss(a_ptr + 2);
            __m256 a3 = _mm256_broadcast_ss(a_ptr + 3);

#if defined(__FMA__)
            c[i][0] = _mm256_fmadd_ps(a0, b0_0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a0, b0_1, c[i][1]);
            c[i][0] = _mm256_fmadd_ps(a1, b1_0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a1, b1_1, c[i][1]);
            c[i][0] = _mm256_fmadd_ps(a2, b2_0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a2, b2_1, c[i][1]);
            c[i][0] = _mm256_fmadd_ps(a3, b3_0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a3, b3_1, c[i][1]);
#else
            c[i][0] = _mm256_add_ps(c[i][0], _mm256_mul_ps(a0, b0_0));
            c[i][1] = _mm256_add_ps(c[i][1], _mm256_mul_ps(a0, b0_1));
            c[i][0] = _mm256_add_ps(c[i][0], _mm256_mul_ps(a1, b1_0));
            c[i][1] = _mm256_add_ps(c[i][1], _mm256_mul_ps(a1, b1_1));
            c[i][0] = _mm256_add_ps(c[i][0], _mm256_mul_ps(a2, b2_0));
            c[i][1] = _mm256_add_ps(c[i][1], _mm256_mul_ps(a2, b2_1));
            c[i][0] = _mm256_add_ps(c[i][0], _mm256_mul_ps(a3, b3_0));
            c[i][1] = _mm256_add_ps(c[i][1], _mm256_mul_ps(a3, b3_1));
#endif
        }
    }

    // 处理剩余的 k
    for (; p < k_size; ++p) {
        const float* bp = B_packed + p * 16;
        __m256 b0 = _mm256_load_ps(bp + 0);
        __m256 b1 = _mm256_load_ps(bp + 8);
        for (int i = 0; i < 6; ++i) {
            __m256 a_val = _mm256_broadcast_ss(A + i * lda + p);
#if defined(__FMA__)
            c[i][0] = _mm256_fmadd_ps(a_val, b0, c[i][0]);
            c[i][1] = _mm256_fmadd_ps(a_val, b1, c[i][1]);
#else
            c[i][0] = _mm256_add_ps(c[i][0], _mm256_mul_ps(a_val, b0));
            c[i][1] = _mm256_add_ps(c[i][1], _mm256_mul_ps(a_val, b1));
#endif
        }
    }

    // 存回 C
    for (int i = 0; i < 6; ++i) {
        _mm256_storeu_ps(C + i * ldc + 0, c[i][0]);
        _mm256_storeu_ps(C + i * ldc + 8, c[i][1]);
    }
}

// 标量边缘处理
void scalar_kernel(int M, int N, int K, const float* A, int lda, const float* B, int ldb, float* C, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int p = 0; p < K; ++p) {
                sum += A[i * lda + p] * B[p * ldb + j];
            }
            C[i * ldc + j] += sum;
        }
    }
}

void gemm_avx_blocked(const float* A, const float* B_packed, const float* B_origin, float* C, int N, int K, int M, int block=128) {
    std::memset(C, 0, sizeof(float) * static_cast<size_t>(N) * M);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int jb = 0; jb < M; jb += block) {
        for (int ib = 0; ib < N; ib += block) {
            int j_end = std::min(jb + block, M);
            int i_end = std::min(ib + block, N);

            // K 维分块
            for (int kb = 0; kb < K; kb += block) {
                int k_end = std::min(kb + block, K);
                int k_len = k_end - kb;

                // Tiling inside block
                for (int j = jb; j < j_end; j += 16) {
                    // 计算 B_packed 中的偏移
                    // B_packed 布局: [Block 0 (16 cols)][Block 1]...
                    // Block j/16 starts at (j/16) * K * 16
                    // Inside block, we need row kb
                    const float* b_ptr = B_packed + (j / 16) * K * 16 + kb * 16;
                    
                    for (int i = ib; i < i_end; i += 6) {
                        if (i + 6 <= i_end && j + 16 <= j_end) {
                            avx_micro_kernel_6x16_packed(k_len, 
                                                         A + i * K + kb, K, 
                                                         b_ptr, 
                                                         C + i * M + j, M);
                        } else {
                            // Edge cases use original B
                            int m_edge = std::min(6, i_end - i);
                            int n_edge = std::min(16, j_end - j);
                            scalar_kernel(m_edge, n_edge, k_len,
                                          A + i * K + kb, K,
                                          B_origin + kb * M + j, M,
                                          C + i * M + j, M);
                        }
                    }
                }
            }
        }
    }
}

float res(const float* C, int N) {
    // 用omp计算每一行的最大值中的最小值
    std::vector<float> row_max(N, -std::numeric_limits<float>::infinity());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        float local_max = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < N; ++j) {
            if (C[i * N + j] > local_max) {
                local_max = C[i * N + j];
            }
        }
        row_max[i] = local_max;
    }
    
    float result = *std::min_element(row_max.begin(), row_max.end());
    return result;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " N seed" << std::endl;
        return 1;
    }
    int N = std::atoi(argv[1]);
    float seed = std::atof(argv[2]);
    if (N <= 0) return 1;

    const size_t align = 64;
    float* A = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * static_cast<size_t>(N) * N));
    float* B = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * static_cast<size_t>(N) * N));
    float* C = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * static_cast<size_t>(N) * N));
    // B_packed 需要稍微大一点以处理 padding
    size_t b_packed_size = sizeof(float) * static_cast<size_t>(N) * N + 4096;
    if (N % 16 != 0) b_packed_size += sizeof(float) * N * 16; 
    float* B_packed = static_cast<float*>(aligned_alloc_helper(align, b_packed_size));

    if (!A || !B || !C || !B_packed) { std::cerr << "alloc failed" << std::endl; return 1; }

    matrix_gen(A, B, N, seed);

#ifdef _OPENMP
    omp_set_num_threads(8);
#endif

    // 全局打包 B 
    pack_B_global(B_packed, B, N, N);

    // auto t0 = std::chrono::high_resolution_clock::now();
    gemm_avx_blocked(A, B_packed, B, C, N, N, N, 128);
    // auto t1 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> dur = t1 - t0;
    // float tr = Trace(C, N, N);
    // std::cout << "Time(ms): " << dur.count() << " Trace: " << tr << std::endl;
    std::cout << res(C, N) << std::endl;
    free(A); free(B); free(C); free(B_packed);
    return 0;
}
