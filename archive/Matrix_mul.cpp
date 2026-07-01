#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <vector>
#include <immintrin.h>
#include <omp.h>

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

float rand_float(float s) {
    return 4 * s * (1 - s);
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


// 基于您的测试结果，最优分块大小为128x128
constexpr int OPTIMAL_BLOCK_SIZE = 128;
constexpr int AVX_REGISTER_COUNT = 16; // 16个YMM寄存器
constexpr int AVX_FLOATS_PER_REG = 8;  // 每个YMM寄存器存8个float

// 专用的AVX微内核 - 针对128x128分块优化
void avx_micro_kernel_6x16(const float* A, const float* B, float* C, 
                          int k, int ldc, int prefetch_offset = 64) {
    // 更可靠但实现清晰的 AVX 微内核：对每个 K 的 8 元素段做向量化累加
    // 假设 A 指向 6 行 (每行 长度 k)，B 指向 16 行（每行长度 k，来自 B_transposed），C 是以 ldc 为行跨度的结果块
    __m256 acc[6][2]; // 每行两组，每组累加 8 列（共 16 列）
    for (int i = 0; i < 6; ++i) {
        acc[i][0] = _mm256_setzero_ps();
        acc[i][1] = _mm256_setzero_ps();
    }

    int p = 0;
    for (; p + 8 <= k; p += 8) {
        // 预取
        _mm_prefetch((const char*)(A + p), _MM_HINT_T0);
        _mm_prefetch((const char*)(B + p), _MM_HINT_T0);

        // 对 A 的 6 行分别加载 8 个连续元素
        __m256 a_vec[6];
        for (int i = 0; i < 6; ++i) {
            a_vec[i] = _mm256_loadu_ps(A + i * k + p);
        }

        // 对 B 的 16 行，每行在偏移 p 处加载 8 个连续元素，并与每行的 a_vec 做点乘累加到对应的 acc
        for (int j = 0; j < 16; ++j) {
            __m256 b_vec = _mm256_loadu_ps(B + j * k + p);
            int block = j < 8 ? 0 : 1;
            for (int i = 0; i < 6; ++i) {
                acc[i][block] = _mm256_add_ps(acc[i][block], _mm256_mul_ps(a_vec[i], b_vec));
            }
        }
    }

    // 处理尾部 (k % 8)
    for (; p < k; ++p) {
        for (int i = 0; i < 6; ++i) {
            float a_scalar = A[i * k + p];
            __m256 a_bcast = _mm256_set1_ps(a_scalar);
            for (int j = 0; j < 16; ++j) {
                float b_scalar = B[j * k + p];
                __m256 b_bcast = _mm256_set1_ps(b_scalar);
                int block = j < 8 ? 0 : 1;
                acc[i][block] = _mm256_add_ps(acc[i][block], _mm256_mul_ps(a_bcast, b_bcast));
            }
        }
    }

    // 将每个 acc[i][block]（包含 8 个列的累加值）写回到 C 矩阵
    for (int i = 0; i < 6; ++i) {
        // 第一组 0..7 列
        _mm256_storeu_ps(C + i * ldc + 0, acc[i][0]);
        // 第二组 8..15 列
        _mm256_storeu_ps(C + i * ldc + 8, acc[i][1]);
    }
}

// AVX 微内核（针对已打包的 B 矩阵） - B_packed 布局为 (k x 16), 每个 p 下有 16 连续列
void avx_micro_kernel_6x16_packed(const float* A, const float* B_packed, float* C, int k, int ldc) {
    __m256 acc[6][2];
    for (int i = 0; i < 6; ++i) {
        acc[i][0] = _mm256_setzero_ps();
        acc[i][1] = _mm256_setzero_ps();
    }

    int p = 0;
    // Unroll the k-loop by 4 to increase ILP and allow multiple fmadd per iteration
    for (; p + 3 < k; p += 4) {
        const float* bp0 = B_packed + static_cast<size_t>(p + 0) * 16;
        const float* bp1 = B_packed + static_cast<size_t>(p + 1) * 16;
        const float* bp2 = B_packed + static_cast<size_t>(p + 2) * 16;
        const float* bp3 = B_packed + static_cast<size_t>(p + 3) * 16;

        __m256 b0_0 = _mm256_load_ps(bp0 + 0);
        __m256 b0_1 = _mm256_load_ps(bp0 + 8);
        __m256 b1_0 = _mm256_load_ps(bp1 + 0);
        __m256 b1_1 = _mm256_load_ps(bp1 + 8);
        __m256 b2_0 = _mm256_load_ps(bp2 + 0);
        __m256 b2_1 = _mm256_load_ps(bp2 + 8);
        __m256 b3_0 = _mm256_load_ps(bp3 + 0);
        __m256 b3_1 = _mm256_load_ps(bp3 + 8);

        for (int i = 0; i < 6; ++i) {
            const float* a_ptr = A + static_cast<size_t>(i) * k;
            __m256 a0 = _mm256_set1_ps(a_ptr[p + 0]);
            __m256 a1 = _mm256_set1_ps(a_ptr[p + 1]);
            __m256 a2 = _mm256_set1_ps(a_ptr[p + 2]);
            __m256 a3 = _mm256_set1_ps(a_ptr[p + 3]);

#if defined(__FMA__)
            acc[i][0] = _mm256_fmadd_ps(a0, b0_0, acc[i][0]);
            acc[i][1] = _mm256_fmadd_ps(a0, b0_1, acc[i][1]);
            acc[i][0] = _mm256_fmadd_ps(a1, b1_0, acc[i][0]);
            acc[i][1] = _mm256_fmadd_ps(a1, b1_1, acc[i][1]);
            acc[i][0] = _mm256_fmadd_ps(a2, b2_0, acc[i][0]);
            acc[i][1] = _mm256_fmadd_ps(a2, b2_1, acc[i][1]);
            acc[i][0] = _mm256_fmadd_ps(a3, b3_0, acc[i][0]);
            acc[i][1] = _mm256_fmadd_ps(a3, b3_1, acc[i][1]);
#else
            acc[i][0] = _mm256_add_ps(acc[i][0], _mm256_mul_ps(a0, b0_0));
            acc[i][1] = _mm256_add_ps(acc[i][1], _mm256_mul_ps(a0, b0_1));
            acc[i][0] = _mm256_add_ps(acc[i][0], _mm256_mul_ps(a1, b1_0));
            acc[i][1] = _mm256_add_ps(acc[i][1], _mm256_mul_ps(a1, b1_1));
            acc[i][0] = _mm256_add_ps(acc[i][0], _mm256_mul_ps(a2, b2_0));
            acc[i][1] = _mm256_add_ps(acc[i][1], _mm256_mul_ps(a2, b2_1));
            acc[i][0] = _mm256_add_ps(acc[i][0], _mm256_mul_ps(a3, b3_0));
            acc[i][1] = _mm256_add_ps(acc[i][1], _mm256_mul_ps(a3, b3_1));
#endif
        }
    }

    // Remainder
    for (; p < k; ++p) {
        const float* bp = B_packed + static_cast<size_t>(p) * 16;
        __m256 b0 = _mm256_load_ps(bp + 0);
        __m256 b1 = _mm256_load_ps(bp + 8);
        for (int i = 0; i < 6; ++i) {
            __m256 a_bcast = _mm256_set1_ps(A[i * k + p]);
#if defined(__FMA__)
            acc[i][0] = _mm256_fmadd_ps(a_bcast, b0, acc[i][0]);
            acc[i][1] = _mm256_fmadd_ps(a_bcast, b1, acc[i][1]);
#else
            acc[i][0] = _mm256_add_ps(acc[i][0], _mm256_mul_ps(a_bcast, b0));
            acc[i][1] = _mm256_add_ps(acc[i][1], _mm256_mul_ps(a_bcast, b1));
#endif
        }
    }

    for (int i = 0; i < 6; ++i) {
        _mm256_storeu_ps(C + i * ldc + 0, acc[i][0]);
        _mm256_storeu_ps(C + i * ldc + 8, acc[i][1]);
    }
}

// 针对缓存冲突优化的分块转置
void cache_optimized_transpose(float* dst, const float* src, int rows, int cols) {
    constexpr int cache_line_size = 64; // 64字节缓存行
    constexpr int floats_per_line = cache_line_size / sizeof(float);
    
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
                                   float* C, int r, int k, int c, int block_size = OPTIMAL_BLOCK_SIZE) {
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

    // 使用AVX微内核执行块乘（B 已转置为 c x k 布局）
    // 主循环按块遍历：对每个 block 内再按 micro-kernel 的大小步进
    // micro-kernel: 6x16 (6 行 * 16 列)，当遇到边界余数时回退到标量实现








        // 简化的并行遍历，直接调用原始 avx_micro_kernel_6x16
    #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int ib = 0; ib < r; ib += block_size) {
            for (int jb = 0; jb < c; jb += block_size) {
                int i_end = std::min(ib + block_size, r);
                int j_end = std::min(jb + block_size, c);

                for (int i0 = ib; i0 < i_end; i0 += 6) {
                    int im = std::min(6, i_end - i0);
                    for (int j0 = jb; j0 < j_end; j0 += 16) {
                        int jn = std::min(16, j_end - j0);

                        if (im == 6 && jn == 16) {
                            const float* A_block = A + i0 * k;
                            const float* B_block = B_transposed + j0 * k;
                            float* C_block = C + i0 * c + j0;

                            _mm_prefetch((const char*)(A_block), _MM_HINT_T0);
                            _mm_prefetch((const char*)(B_block), _MM_HINT_T0);

                            avx_micro_kernel_6x16(A_block, B_block, C_block, k, c, 64);
                        } else {
                            for (int ii = 0; ii < im; ++ii) {
                                for (int jj = 0; jj < jn; ++jj) {
                                    float sum = 0.0f;
                                    const float* a_row = A + (i0 + ii) * k;
                                    const float* b_row = B_transposed + (j0 + jj) * k;
                                    for (int p = 0; p < k; ++p) {
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
    free(B_transposed);
    return C;
}

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

void test_block(int N,int K,int M,float seed, int start, int end, int step) {
    float *A= static_cast<float*>(aligned_alloc_helper(64, sizeof(float)*N*K));
    float *B= static_cast<float*>(aligned_alloc_helper(64, sizeof(float)*K*M));
    matrix_gen(A, B, N, seed);
    float *C= static_cast<float*>(aligned_alloc_helper(64, sizeof(float)*N*M));
    for (int i = start; i < end; i += step) {
        auto start = std::chrono::high_resolution_clock::now();
        optimized_matrix_mul_xeon_e5(A, B, C, N, K, M, i);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        float trace = Trace(C, N, M);
        std::cout << "Size: " << N << "x" << M << ", Time: " << duration.count() << " ms, Trace: " << trace << std::endl;
    }

    // float *D = static_cast<float*>(aligned_alloc_helper(64, sizeof(float)*N*M));
    // base_mul(D, A, B, N, K, M);
    // for (int i = 0; i < N*M; ++i) {
    //     if (std::abs(C[i] - D[i]) > 1e-3) {
    //         std::cerr << "Mismatch at index " << i << ": C=" << C[i] << ", D=" << D[i] << std::endl;
    //         break;
    //     }
    // }
    // free(D);

    free(A);
    free(B);
    free(C);
}

int main(int argc, char** argv) {
    int N = 1024;
    float seed = 0.0f;
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " N seed" << std::endl;
        std::cout << "Using fault\n";
        return 1;
    }
    N = std::atoi(argv[1]);
    seed = std::atof(argv[2]);
    if (N <= 0) {
        std::cerr << "Matrix size must be positive." << std::endl;
        return 1;
    }
    if (seed <= 0.0f || seed >= 1.0f) {
        std::cerr << "Seed must be in the range (0, 1)." << std::endl;
        return 1;
    }
    auto A = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
    auto B = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
    auto C = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
    if (!A || !B || !C) {
        std::cerr << "Memory allocation failed." << std::endl;
        free(A);
        free(B);
        free(C);
        return 1;
    }
    matrix_gen(A, B, N, seed);
    #ifdef TEST
    auto start = std::chrono::high_resolution_clock::now();
    optimized_matrix_mul_xeon_e5(A, B, C, N, N, N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    std::cout << "Time: " << duration.count() << " ms" << std::endl;
    #else
    optimized_matrix_mul_xeon_e5(A, B, C, N, N, N);
    #endif
    float trace = Trace(C, N, N);
    std::cout << trace << std::endl;
    free(A);
    free(B);
    free(C);
    // std::cout << "seed:";
    // std::cin >> seed;
    // int start, end, step;
    // std::cout << "block size start, end, step:";
    // std::cin >> start >> end >> step;
    // test_block(N, N, N, seed, start, end, step);
    return 0;
}