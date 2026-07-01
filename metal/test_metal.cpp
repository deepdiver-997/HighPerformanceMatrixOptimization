// =============================================================================
// Metal GPU 矩阵乘法 — 性能对比测试
//
// 这个文件是一个纯 C++ (.cpp) 的测试程序。
// 它通过 metal_matmul.h 调用 GPU 加速，但完全不需要知道 ObjC/Metal 的存在。
//
// 【测试对比】
//   ┌──────────────────┬───────────────────────┬──────────────┐
//   │ 实现              │ 位置                   │ 说明          │
//   ├──────────────────┼───────────────────────┼──────────────┤
//   │ CPU Naive        │ matrix.cpp: base_mul  │ 三重循环基准  │
//   │ CPU Best         │ matrix.cpp: async_*   │ SIMD+多线程   │
//   │ GPU Naive        │ metal_shaders.metal   │ 每个线程一个  │
//   │ GPU Tiled        │ metal_shaders.metal   │ threadgroup分块│
//   └──────────────────┴───────────────────────┴──────────────┘
//
// 【为什么不是 .mm 文件？】
//   这个文件只 include "metal_matmul.h" (纯 C++ 头文件)。
//   所有 ObjC 调用都在 metal_matmul.mm 内部完成。
//   这体现了 .mm 桥接的核心价值: 隔离 ObjC 依赖。
//
// 【编译方式】
//   见 ../metal/Makefile
//   需要同时编译 test_metal.cpp + metal_matmul.mm + ../matrix.cpp
//   链接 Metal.framework + Foundation.framework
// =============================================================================

#include "metal_matmul.h"   // GPU 接口 (纯 C++)
#include "../matrix.h"      // CPU 矩阵运算 (纯 C++)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// ---- 计时工具 ----
struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    void begin() { start = std::chrono::high_resolution_clock::now(); }
    double elapsed_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};

// ---- 正确性验证: 计算矩阵对角线和 (trace) ----
// trace 相同时矩阵不一定完全相同，但 trace 不同则一定不同
// 这是快速验证计算正确性的方法
float compute_trace(const float* C, int N) {
    float t = 0.0f;
    for (int i = 0; i < N; i++) t += C[i * N + i];
    return t;
}

// ---- 性能指标: GFlops = 10^9 次浮点运算/秒 ----
// 矩阵乘法的 FLOP 数 = 2 * M * N * K (每个 C 元素需要 K 次乘法和 K 次加法)
double compute_gflops(int M, int K, int N, double time_ms) {
    return (2.0 * M * K * N) / (time_ms * 1e6);
}

int main(int argc, char** argv) {
    // ====================================================
    // 第0步: 解析命令行参数
    // ====================================================
    int test_sizes[] = {256, 512, 1024, 2048};
    int num_sizes = 4;
    int runs = 3;
    float seed = 0.3f;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            test_sizes[0] = atoi(argv[i + 1]);
            num_sizes = 1;
            i++;
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            runs = atoi(argv[i + 1]);
            i++;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-s size] [-n runs] [-h]\n", argv[0]);
            printf("  -s size   Matrix size (default: 256,512,1024,2048)\n");
            printf("  -n runs   Number of runs per test (default: 3)\n");
            return 0;
        }
    }

    // ====================================================
    // 第1步: 初始化 Metal GPU
    // ====================================================
    printf("=== Initializing Metal GPU ===\n");

    // metal_init 内部: ObjC 代码获取 GPU、编译 shader、创建 pipeline
    // 但 test_metal.cpp 只需要知道返回的 MetalContext* 指针
    MetalContext* metal = metal_init("metal_shaders.metal");
    if (!metal) {
        fprintf(stderr, "Failed to initialize Metal. Exiting.\n");
        return 1;
    }
    printf("GPU: %s\n\n", metal_gpu_name(metal));

    // ====================================================
    // 第2步: CPU vs GPU 对比测试
    // ====================================================
    printf("%-8s | %-18s %-10s %-10s | %-18s %-10s %-10s | %-18s %-10s %-10s\n",
           "Size",
           "CPU Naive", "Time(ms)", "GFlops",
           "CPU Best", "Time(ms)", "GFlops",
           "GPU Naive", "Time(ms)", "GFlops");
    printf("%-8s-+-%s-+-%s\n",
           "--------",
           "------------------------------------------------------------",
           "---------------------------------------");

    for (int si = 0; si < num_sizes; si++) {
        int N = test_sizes[si];

        // 分配 CPU 内存 (64 字节对齐)
        float *A = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
        float *B = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
        float *C_cpu_naive = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
        float *C_cpu_best  = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
        float *C_gpu_naive = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));

        if (!A || !B || !C_cpu_naive || !C_cpu_best || !C_gpu_naive) {
            printf("Memory allocation failed for N=%d\n", N);
            continue;
        }

        // 生成随机测试数据
        matrix_gen(A, B, N, seed);

        double cpu_naive_time = 0, cpu_best_time = 0, gpu_naive_time = 0;

        // ---- 测试 A: CPU 朴素实现 (base_mul) ----
        for (int r = 0; r < runs; r++) {
            memset(C_cpu_naive, 0, sizeof(float) * N * N);
            Timer t; t.begin();
            base_mul(A, B, C_cpu_naive, N, N, N);
            cpu_naive_time += t.elapsed_ms();
        }
        cpu_naive_time /= runs;
        float trace_naive = compute_trace(C_cpu_naive, N);

        // ---- 测试 B: CPU 最佳实现 (SIMD + 多线程) ----
        for (int r = 0; r < runs; r++) {
            memset(C_cpu_best, 0, sizeof(float) * N * N);
            Timer t; t.begin();
            async_matrix_mul_trans_block_with_simd(A, B, C_cpu_best, N, N, N, 64);
            cpu_best_time += t.elapsed_ms();
        }
        cpu_best_time /= runs;
        float trace_best = compute_trace(C_cpu_best, N);

        // ---- 测试 C: GPU 朴素实现 (naive_matmul) ----
        // metal_matmul_naive 内部流程:
        //   CPU → GPU: 拷贝 A, B
        //   GPU: 成千线程并行执行 naive_matmul kernel
        //   GPU → CPU: 拷贝 C 回来
        for (int r = 0; r < runs; r++) {
            memset(C_gpu_naive, 0, sizeof(float) * N * N);
            Timer t; t.begin();
            metal_matmul_naive(metal, A, B, C_gpu_naive, N, N, N);
            gpu_naive_time += t.elapsed_ms();
        }
        gpu_naive_time /= runs;
        float trace_gpu = compute_trace(C_gpu_naive, N);

        // ---- 打印结果 ----
        printf("%-8d | CPU Naive         %8.2f %8.2f | CPU Best          %8.2f %8.2f | GPU Naive         %8.2f %8.2f\n",
               N,
               cpu_naive_time, compute_gflops(N, N, N, cpu_naive_time),
               cpu_best_time,  compute_gflops(N, N, N, cpu_best_time),
               gpu_naive_time, compute_gflops(N, N, N, gpu_naive_time));

        // ---- 验证 GPU 结果正确性 ----
        printf("          Trace: CPU_naive=%.4e  CPU_best=%.4e  GPU_naive=%.4e",
               trace_naive, trace_best, trace_gpu);

        float max_err = 0.0f;
        for (int i = 0; i < N * N; i++) {
            float err = fabsf(C_gpu_naive[i] - C_cpu_naive[i]);
            if (err > max_err) max_err = err;
        }
        printf("  MaxErr=%.2e", max_err);
        if (max_err > 1.0f) {
            printf("  *** WARNING: Large error ***");
        }
        printf("\n");

        free(A); free(B);
        free(C_cpu_naive); free(C_cpu_best); free(C_gpu_naive);
    }

    // ====================================================
    // 第3步: GPU Naive vs GPU Tiled 对比
    // ====================================================
    printf("\n=== GPU Tiled vs GPU Naive Comparison ===\n");
    printf("%-8s | %-18s %-10s %-10s | %-18s %-10s %-10s | %-8s\n",
           "Size", "GPU Naive", "Time(ms)", "GFlops",
           "GPU Tiled", "Time(ms)", "GFlops", "Speedup");

    for (int si = 0; si < num_sizes; si++) {
        int N = test_sizes[si];

        float *A = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
        float *B = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
        float *C_naive = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
        float *C_tiled = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));

        matrix_gen(A, B, N, seed);

        double naive_t = 0, tiled_t = 0;

        for (int r = 0; r < runs; r++) {
            memset(C_naive, 0, sizeof(float) * N * N);
            Timer t; t.begin();
            metal_matmul_naive(metal, A, B, C_naive, N, N, N);
            naive_t += t.elapsed_ms();
        }
        naive_t /= runs;

        for (int r = 0; r < runs; r++) {
            memset(C_tiled, 0, sizeof(float) * N * N);
            Timer t; t.begin();
            metal_matmul_tiled(metal, A, B, C_tiled, N, N, N);
            tiled_t += t.elapsed_ms();
        }
        tiled_t /= runs;

        double speedup = naive_t / tiled_t;

        // 验证两个 GPU 实现输出一致
        float max_err = 0.0f;
        for (int i = 0; i < N * N; i++) {
            float err = fabsf(C_tiled[i] - C_naive[i]);
            if (err > max_err) max_err = err;
        }

        printf("%-8d | GPU Naive         %8.2f %8.2f | GPU Tiled         %8.2f %8.2f | %5.2fx  err=%.1e\n",
               N,
               naive_t, compute_gflops(N, N, N, naive_t),
               tiled_t,  compute_gflops(N, N, N, tiled_t),
               speedup, max_err);

        free(A); free(B); free(C_naive); free(C_tiled);
    }

    // ====================================================
    // 第4步: GPU Tiled vs GPU Simd 对比
    // ====================================================
    printf("\n=== GPU Simd vs GPU Tiled Comparison ===\n");
    printf("%-8s | %-18s %-10s %-10s | %-18s %-10s %-10s | %-8s\n",
           "Size", "GPU Tiled", "Time(ms)", "GFlops",
           "GPU Simd", "Time(ms)", "GFlops", "Speedup");

    for (int si = 0; si < num_sizes; si++) {
        int N = test_sizes[si];

        float *A = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
        float *B = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
        float *C_tiled = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
        float *C_simd  = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));

        matrix_gen(A, B, N, seed);

        double tiled_t = 0, simd_t = 0;

        for (int r = 0; r < runs; r++) {
            memset(C_tiled, 0, sizeof(float) * N * N);
            Timer t; t.begin();
            metal_matmul_tiled(metal, A, B, C_tiled, N, N, N);
            tiled_t += t.elapsed_ms();
        }
        tiled_t /= runs;

        for (int r = 0; r < runs; r++) {
            memset(C_simd, 0, sizeof(float) * N * N);
            Timer t; t.begin();
            metal_matmul_simd(metal, A, B, C_simd, N, N, N);
            simd_t += t.elapsed_ms();
        }
        simd_t /= runs;

        double speedup = tiled_t / simd_t;

        float max_err = 0.0f;
        for (int i = 0; i < N * N; i++) {
            float err = fabsf(C_simd[i] - C_tiled[i]);
            if (err > max_err) max_err = err;
        }

        printf("%-8d | GPU Tiled         %8.2f %8.2f | GPU Simd          %8.2f %8.2f | %5.2fx  err=%.1e\n",
               N,
               tiled_t, compute_gflops(N, N, N, tiled_t),
               simd_t,  compute_gflops(N, N, N, simd_t),
               speedup, max_err);

        free(A); free(B); free(C_tiled); free(C_simd);
    }

    // ====================================================
    // 第5步: 清理
    // ====================================================
    //
    // metal_release 内部:
    //   delete ctx → 析构 C++ struct
    //   ObjC 成员 (id<MTLDevice> 等) 由 ARC 自动 release
    //
    metal_release(metal);
    printf("\nDone.\n");
    return 0;
}
