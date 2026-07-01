// =============================================================================
// matmul.cpp — 矩阵乘法统一实现
//
// 包含:
//   1. CPU 路径 (base / block / SIMD / SIMD+threads)
//   2. GPU 桥接 (通过 metal/metal_matmul.h)
//   3. matmul() 自动调度
// =============================================================================

#include "matmul.h"
#include "matmul_arch.h"
#include "pool.h"
#include "getCacheSize.h"

#include <cstring>     // memset
#include <cstdlib>     // posix_memalign
#include <algorithm>   // std::min
#include <cmath>       // std::sqrt
#include <vector>      // std::vector
#include <future>      // std::future
#include <thread>      // std::thread::hardware_concurrency

// =============================================================================
// 工具函数
// =============================================================================

static void* aligned_alloc_64(size_t size) {
    size_t sz = ((size + 63) / 64) * 64;
    void* ptr = nullptr;
#if __STDC_VERSION__ >= 201112L
    ptr = aligned_alloc(64, sz);
#else
    posix_memalign(&ptr, 64, sz);
#endif
    return ptr;
}

// =============================================================================
// 第1层: CPU 朴素实现 (三重循环, 永远可用)
// =============================================================================

static void cpu_naive(const float* A, const float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int p = 0; p < K; p++) {
                sum += A[i * K + p] * B[p * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// =============================================================================
// 第2层: CPU 分块 + 转置 (标量, 无 SIMD)
// =============================================================================

// 缓存优化的转置: 按 cache-line 大小分块转置，减少冲突 miss
static void transpose_blocked(float* dst, const float* src, int rows, int cols) {
    const int T = 16;  // ~64 bytes per row of tile
    for (int i = 0; i < rows; i += T) {
        for (int j = 0; j < cols; j += T) {
            int ie = (i + T < rows) ? i + T : rows;
            int je = (j + T < cols) ? j + T : cols;
            for (int ii = i; ii < ie; ii++) {
                for (int jj = j; jj < je; jj++) {
                    dst[jj * rows + ii] = src[ii * cols + jj];
                }
            }
        }
    }
}

static void cpu_block_trans(const float* A, const float* B, float* C,
                             int M, int K, int N, int block_size) {
    // 转置 B → Bt (K×N → N×K)
    float* Bt = static_cast<float*>(aligned_alloc_64(sizeof(float) * K * N));
    transpose_blocked(Bt, B, K, N);  // Bt is N×K

    for (int i0 = 0; i0 < M; i0 += block_size) {
        int i_end = (i0 + block_size < M) ? i0 + block_size : M;
        for (int j0 = 0; j0 < N; j0 += block_size) {
            int j_end = (j0 + block_size < N) ? j0 + block_size : N;

            for (int i = i0; i < i_end; i++) {
                for (int j = j0; j < j_end; j++) {
                    float sum = 0.0f;
                    for (int p = 0; p < K; p++) {
                        sum += A[i * K + p] * Bt[j * K + p];  // Bt[j][p] = B[p][j]
                    }
                    C[i * N + j] = sum;
                }
            }
        }
    }
    free(Bt);
}

// =============================================================================
// 第3层: CPU SIMD (NEON / AVX / SSE, 编译期选定)
//
// 策略: 分块 + 转置 + 内层 SIMD 向量化
// 利用 matmul_simd 命名空中间的 SIMD 操作抹平平台差异
// =============================================================================

static void cpu_simd(const float* A, const float* B, float* C,
                      int M, int K, int N, int block_size) {
    constexpr int W = MATMUL_SIMD_WIDTH;  // 4 (NEON/SSE) or 8 (AVX)

    // 转置 B
    float* Bt = static_cast<float*>(aligned_alloc_64(sizeof(float) * K * N));
    transpose_blocked(Bt, B, K, N);

    for (int i0 = 0; i0 < M; i0 += block_size) {
        int i_end = (i0 + block_size < M) ? i0 + block_size : M;
        for (int j0 = 0; j0 < N; j0 += block_size) {
            int j_end = (j0 + block_size < N) ? j0 + block_size : N;

            for (int i = i0; i < i_end; i++) {
                for (int j = j0; j < j_end; j++) {

                    // 4 个 SIMD 累加器 (循环展开)
                    matmul_f32 c0 = matmul_simd::set1(0.0f);
                    matmul_f32 c1 = matmul_simd::set1(0.0f);
                    matmul_f32 c2 = matmul_simd::set1(0.0f);
                    matmul_f32 c3 = matmul_simd::set1(0.0f);

                    int p = 0;
                    for (; p + 4 * W <= K; p += 4 * W) {
                        matmul_f32 a0 = matmul_simd::set1(A[i * K + p + 0 * W]);
                        matmul_f32 a1 = matmul_simd::set1(A[i * K + p + 1 * W]);
                        matmul_f32 a2 = matmul_simd::set1(A[i * K + p + 2 * W]);
                        matmul_f32 a3 = matmul_simd::set1(A[i * K + p + 3 * W]);

                        // Bt[j][p..] 在内存中是连续的 (因为转置了)
                        c0 = matmul_simd::fmadd(a0, matmul_simd::load(&Bt[j * K + p + 0 * W]), c0);
                        c1 = matmul_simd::fmadd(a1, matmul_simd::load(&Bt[j * K + p + 1 * W]), c1);
                        c2 = matmul_simd::fmadd(a2, matmul_simd::load(&Bt[j * K + p + 2 * W]), c2);
                        c3 = matmul_simd::fmadd(a3, matmul_simd::load(&Bt[j * K + p + 3 * W]), c3);
                    }

                    // 尾部 (不足 4*W 的部分)
                    for (; p + W <= K; p += W) {
                        matmul_f32 a = matmul_simd::set1(A[i * K + p]);
                        c0 = matmul_simd::fmadd(a, matmul_simd::load(&Bt[j * K + p]), c0);
                    }

                    // 把 4 个累加器加起来
                    c0 = matmul_simd::add(c0, c1);
                    c2 = matmul_simd::add(c2, c3);
                    c0 = matmul_simd::add(c0, c2);

                    float sum = matmul_simd::horizontal_sum(c0);

                    // 标量尾部
                    for (; p < K; p++) {
                        sum += A[i * K + p] * Bt[j * K + p];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
    }
    free(Bt);
}

// =============================================================================
// 第4层: CPU SIMD + 多线程
//
// 行级并行: 把 M 行分给多个线程，每个线程独立计算自己那部分 C
// =============================================================================

static void cpu_simd_threaded(const float* A, const float* B, float* C,
                               int M, int K, int N, int block_size) {
    ThreadPool& pool = ThreadPool::get_instance();
    int num_threads = (int)std::thread::hardware_concurrency();

    if (num_threads <= 1 || M < 128) {
        cpu_simd(A, B, C, M, K, N, block_size);
        return;
    }

    int rows_per_thread = (M + num_threads - 1) / num_threads;
    std::vector<std::future<void>> futures;

    for (int t = 0; t < num_threads; t++) {
        int row_start = t * rows_per_thread;
        int row_end = (row_start + rows_per_thread < M) ? row_start + rows_per_thread : M;
        if (row_start >= M) break;

        futures.push_back(pool.enqueue_task([=]() {
            cpu_simd(A + row_start * K, B, C + row_start * N,
                      row_end - row_start, K, N, block_size);
        }));
    }

    for (auto& f : futures) f.get();
}

// =============================================================================
// 第5层: CPU 小矩阵专用 (避免不必要的内存分配和线程开销)
// =============================================================================

static void cpu_small(const float* A, const float* B, float* C, int M, int K, int N) {
    // 直接 SIMD 单线程，不做转置 (省 alloc 开销)
    constexpr int W = MATMUL_SIMD_WIDTH;
    constexpr int SMALL_LIMIT = 128;

    if (M > SMALL_LIMIT || N > SMALL_LIMIT) {
        cpu_simd(A, B, C, M, K, N, 32);
        return;
    }

    // 非常小的矩阵: 直接三重循环 + SIMD 内层
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            matmul_f32 c = matmul_simd::set1(0.0f);
            int p = 0;
            for (; p + W <= K; p += W) {
                matmul_f32 a = matmul_simd::set1(A[i * K + p]);
                // B 的列访问是跳步的，但对于小矩阵可以接受
                float b_buf[W];
                for (int w = 0; w < W; w++) b_buf[w] = B[(p + w) * N + j];
                c = matmul_simd::fmadd(a, matmul_simd::load(b_buf), c);
            }
            float sum = matmul_simd::horizontal_sum(c);
            for (; p < K; p++) sum += A[i * K + p] * B[p * N + j];
            C[i * N + j] = sum;
        }
    }
}

// =============================================================================
// Accelerate 路径 (Apple AMX 协处理器)
// =============================================================================

#ifdef MATMUL_HAS_ACCELERATE
#define ACCELERATE_NEW_LAPACK 1
#define ACCELERATE_LAPACK_ILP64 1
#include <Accelerate/Accelerate.h>

static void cpu_accelerate(const float* A, const float* B, float* C,
                            int M, int K, int N) {
    // cblas_sgemm 使用 AMX + NEON + 最优分块
    // C = 1.0 * A * B + 0.0 * C  (row-major → column-major 转换)
    // 因为 BLAS 是 column-major, 所以参数顺序是 B, A
    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K, B, N,
                0.0f, C, N);
}
#endif

// =============================================================================
// GPU 路径 (Metal)
// =============================================================================

#ifdef MATMUL_HAS_METAL
#include "metal/metal_matmul.h"

// ---- 全局 GPU 上下文 (惰性初始化) ----
static MetalContext* g_gpu_ctx = nullptr;

bool matmul_gpu_available() {
    return g_gpu_ctx != nullptr;
}

bool matmul_gpu_init(const char* metallib_path) {
    if (g_gpu_ctx) return true;
    g_gpu_ctx = metal_init(metallib_path);
    return g_gpu_ctx != nullptr;
}

void matmul_gpu_release() {
    if (g_gpu_ctx) {
        metal_release(g_gpu_ctx);
        g_gpu_ctx = nullptr;
    }
}

void matmul_gpu(const float* A, const float* B, float* C, int M, int K, int N) {
    if (!g_gpu_ctx) {
        // 回退: GPU 不可用时走 CPU 路径
        matmul_cpu(A, B, C, M, K, N);
        return;
    }
    // 使用 simdgroup kernel (性能最好)
    metal_matmul_simd(g_gpu_ctx, A, B, C, M, K, N);
}

#endif // MATMUL_HAS_METAL

// =============================================================================
// CPU 最优路径 (编译期选择)
// =============================================================================

void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N) {
    if (M <= 0 || K <= 0 || N <= 0) return;
    memset(C, 0, sizeof(float) * M * N);

    // 小矩阵阈值: 总运算量 < 1M FLOP → 不值得优化
    const int total_elements = M * K + K * N + M * N;
    if (total_elements < 64 * 64 * 3) {
        cpu_naive(A, B, C, M, K, N);
        return;
    }

    // 动态选分块大小 (基于 L1 cache)
    int l1_size = (int)get_L1d_cache_size();
    // 块大小 ≈ sqrt(L1 / 2 / sizeof(float)) 的一半
    // 两个矩阵各占一半 L1
    int block_size = (int)std::sqrt((float)(l1_size / 2) / sizeof(float));
    block_size = std::max(32, std::min(256, block_size));
    block_size = (block_size / MATMUL_SIMD_WIDTH) * MATMUL_SIMD_WIDTH;

#ifdef MATMUL_HAS_ACCELERATE
    // Apple AMX: 直接甩给 Accelerate (这是最快的 CPU 路径)
    cpu_accelerate(A, B, C, M, K, N);
#else
    // ARM NEON / x86 AVX / x86 SSE / scalar:
    cpu_simd_threaded(A, B, C, M, K, N, block_size);
#endif
}

// =============================================================================
// ★ 主入口: 自动选择最优路径
// =============================================================================

void matmul(const float* A, const float* B, float* C, int M, int K, int N) {
    if (M <= 0 || K <= 0 || N <= 0) return;

    const long long total_ops = (long long)M * K * N;

    // 极小矩阵: 直接朴素，任何优化都是过度设计
    if (total_ops <= 128LL * 128 * 128) {
        cpu_naive(A, B, C, M, K, N);
        return;
    }

#ifdef MATMUL_HAS_ACCELERATE
    // Apple Accelerate (AMX): 对所有大小都极快，不需 GPU
    return matmul_cpu(A, B, C, M, K, N);
#endif

#ifdef MATMUL_HAS_METAL
    // 无 Accelerate 但 GPU 可用:
    //   小矩阵 → CPU (GPU 启动开销 > 计算收益)
    //   大矩阵 → GPU
    if (g_gpu_ctx && total_ops > 1024LL * 1024 * 1024) {
        return matmul_gpu(A, B, C, M, K, N);
    }
#endif

    // 回退: CPU SIMD+多线程
    return matmul_cpu(A, B, C, M, K, N);
}
