// 针对Xeon E5-2609 v2的专用优化矩阵乘法
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cstddef>
#include <cstring>
#include <vector>
#include <immintrin.h>
#include <omp.h>

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
                                   float* C, int r, int k, int c) {
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

// 增强的测试函数，用于验证不同分块大小的性能
// void benchmark_optimal_block_size(const float* A, const float* B, 
//                                  int r, int k, int c, float seed) {
//     std::vector<int> block_sizes = {128, 136, 144, 152, 160, 192, 256};
//     double best_time = std::numeric_limits<double>::max();
//     int best_block_size = OPTIMAL_BLOCK_SIZE;
    
//     for (int block_size : block_sizes) {
//         auto start = std::chrono::high_resolution_clock::now();
        
//         // 临时修改分块大小进行测试
//         float* C = optimized_matrix_mul_xeon_e5_custom_block(A, B, nullptr, r, k, c, block_size);
        
//         auto end = std::chrono::high_resolution_clock::now();
//         double duration = std::chrono::duration<double, std::milli>(end - start).count();
        
//         std::cout << "Block size " << block_size << ": " << duration << " ms" << std::endl;
        
//         if (duration < best_time) {
//             best_time = duration;
//             best_block_size = block_size;
//         }
        
//         free(C);
//     }
    
//     std::cout << "Optimal block size: " << best_block_size << " with " << best_time << " ms" << std::endl;
// }

// 缓存行对齐的内存分配
void* cache_aligned_alloc(size_t size, size_t alignment = 64) {
    void* ptr = aligned_alloc(alignment, size);
    if (ptr) {
        // 确保内存按缓存行对齐
        memset(ptr, 0, size);
    }
    return ptr;
}

// 缓存冲突避免策略
class CacheConflictAvoidance {
private:
    static constexpr int CACHE_WAYS = 16; // E5-2609 v2是16路关联
    static constexpr int CACHE_LINE_SIZE = 64;
    
public:
    // 计算避免缓存冲突的偏移量
    static int get_cache_conflict_offset(int base_size) {
        // 为不同的OpenMP线程分配不同的缓存集
    int thread_id = 0;
#ifdef _OPENMP
    thread_id = omp_get_thread_num();
#endif
    return (thread_id * (CACHE_LINE_SIZE / sizeof(float))) % base_size;
    }
    
    // 缓存友好的内存访问模式
    static void cache_friendly_access_pattern(float* data, int size) {
        int stride = CACHE_LINE_SIZE / sizeof(float);
        for (int i = 0; i < size; i += stride) {
            // 以缓存行大小为单位进行访问
            _mm_prefetch((char*)(data + i), _MM_HINT_T0);
        }
    }
};

// 针对缓存关联性的优化分块
// void cache_associative_optimized_mul(const float* A, const float* B, float* C, 
//                                     int r, int k, int c) {
//     const int cache_ways = 16; // 16路关联缓存
//     const int cache_line_size = 64;
    
//     // 计算避免冲突的分块大小
//     int optimal_mr = 6;  // 微内核行数
//     int optimal_nr = 16; // 微内核列数
    
//     // 主分块循环
//     #pragma omp parallel for collapse(2)
//     for (int i = 0; i < r; i += optimal_mr) {
//         for (int j = 0; j < c; j += optimal_nr) {
//             // 为每个线程计算不同的缓存偏移
//             int cache_offset = CacheConflictAvoidance::get_cache_conflict_offset(optimal_mr * optimal_nr);
            
//             int i_end = std::min(i + optimal_mr, r);
//             int j_end = std::min(j + optimal_nr, c);
            
//             // 应用缓存友好的微内核
//             avx_micro_kernel_6x16_with_cache_offset(
//                 A + i * k, B + j * k, C + i * c + j + cache_offset,
//                 k, c, cache_offset
//             );
//         }
//     }
// }

bool comp(float *a, float *b, int N) {
    for (int i = 0; i < N * N; ++i) {
        if (std::abs(a[i] - b[i]) > 1e-3) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
            return false;
        }
    }
    return true;
}

float Trace(const float* A, int r, int c) {
    float sum = 0.0f;
    int n = std::min(r, c);
    for (int i = 0; i < n; ++i) {
        sum += A[i * c + i];
    }
    return sum;
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

void base_cal(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            C[i * N + j] = 0;
            for (int k = 0; k < N; ++k) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <seed> <size>" << std::endl;
        return -1;
    }
    float seed = std::stof(argv[1]);
    int size = std::stoi(argv[2]);
    float *A = static_cast<float*>(aligned_alloc(64, sizeof(float) * size * size));
    float *B = static_cast<float*>(aligned_alloc(64, sizeof(float) * size * size));
    float *C = static_cast<float*>(aligned_alloc(64, sizeof(float) * size * size));
    float *D = static_cast<float*>(aligned_alloc(64, sizeof(float) * size * size));
    if (!A || !B || !C || !D) {
        std::cerr << "Aligned allocation failed" << std::endl;
        free(A);
        free(B);
        free(C);
        free(D);
        return -1;
    }

    // 生成测试矩阵
    matrix_gen(A, B, size, seed);

    // 进行矩阵乘法
    {
        timer t;
        t.start_timer();
        optimized_matrix_mul_xeon_e5(A, B, C, size, size, size);
        t.end_timer();
    }

    // 验证结果
    {
        base_cal(A, B, D, size);
        if (!comp(C, D, size)) {
            std::cerr << "Result verification failed" << std::endl;
        }
    }

    free(A);
    free(B);
    free(C);
    free(D);
    return 0;
}