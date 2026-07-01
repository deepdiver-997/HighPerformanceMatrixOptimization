// =============================================================================
// test_matmul.cpp — 测试新的统一 matmul 接口
//
// 编译 (Apple + Metal):
//   clang++ -std=c++17 -O3 test_matmul.cpp matmul.cpp metal/metal_matmul.mm \
//           -framework Metal -framework Foundation -framework Accelerate \
//           -pthread -o test_matmul
//
// 编译 (无 GPU, 只有 CPU):
//   clang++ -std=c++17 -O3 test_matmul.cpp matmul.cpp \
//           -framework Accelerate -pthread -o test_matmul
//
// 编译 (通用 ARM Linux):
//   g++ -std=c++17 -O3 test_matmul.cpp matmul.cpp -pthread -o test_matmul
// =============================================================================

#include "matmul.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

// 计时
struct Timer {
    std::chrono::high_resolution_clock::time_point start;
    void begin() { start = std::chrono::high_resolution_clock::now(); }
    double ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    }
};

static double gflops(int M, int K, int N, double ms) {
    return (2.0 * M * K * N) / (ms * 1e6);
}

// 验证 trace
static float trace(const float* C, int N) {
    float t = 0;
    for (int i = 0; i < N; i++) t += C[i * N + i];
    return t;
}

// 生成测试数据
static void gen_matrix(float* m, int N, float seed) {
    for (int i = 0; i < N * N; i++) {
        m[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

int main(int argc, char** argv) {
    int sizes[] = {256, 512, 1024, 2048};
    int num_sizes = 4;
    int runs = 3;

    if (argc > 1) { sizes[0] = atoi(argv[1]); num_sizes = 1; }
    if (argc > 2) runs = atoi(argv[2]);

    // ---- 初始化 GPU (如果可用) ----
    bool gpu_ok = false;
#ifdef MATMUL_HAS_METAL
    printf("Metal GPU: ");
    if (matmul_gpu_init("metal/metal_shaders.metal")) {
        printf("available\n");
        gpu_ok = true;
    } else {
        printf("init failed, using CPU only\n");
    }
#else
    printf("Metal GPU: not compiled in\n");
#endif

    printf("\n");

    // ---- 正确性验证 ----
    printf("=== Correctness Check (256x256) ===\n");
    int N = 256;
    float *A = (float*)malloc(sizeof(float) * N * N);
    float *B = (float*)malloc(sizeof(float) * N * N);
    float *C_cpu = (float*)malloc(sizeof(float) * N * N);
    float *C_auto = (float*)malloc(sizeof(float) * N * N);

    gen_matrix(A, N, 0.5f);
    gen_matrix(B, N, 0.3f);

    memset(C_cpu, 0, sizeof(float) * N * N);
    matmul_cpu(A, B, C_cpu, N, N, N);
    float t_cpu = trace(C_cpu, N);

    memset(C_auto, 0, sizeof(float) * N * N);
    matmul(A, B, C_auto, N, N, N);
    float t_auto = trace(C_auto, N);

    float max_err = 0;
    for (int i = 0; i < N * N; i++)
        if (fabsf(C_auto[i] - C_cpu[i]) > max_err)
            max_err = fabsf(C_auto[i] - C_cpu[i]);

    printf("  CPU trace = %.6e, Auto trace = %.6e, MaxErr = %.2e  %s\n",
           t_cpu, t_auto, max_err, max_err < 1e-3 ? "PASS" : "FAIL");

    if (gpu_ok) {
        float *C_gpu = (float*)malloc(sizeof(float) * N * N);
        memset(C_gpu, 0, sizeof(float) * N * N);
        matmul_gpu(A, B, C_gpu, N, N, N);
        float t_gpu = trace(C_gpu, N);
        float max_err2 = 0;
        for (int i = 0; i < N * N; i++)
            if (fabsf(C_gpu[i] - C_cpu[i]) > max_err2)
                max_err2 = fabsf(C_gpu[i] - C_cpu[i]);
        printf("  GPU trace = %.6e, MaxErr = %.2e  %s\n",
               t_gpu, max_err2, max_err2 < 1e-3 ? "PASS" : "FAIL");
        free(C_gpu);
    }
    free(A); free(B); free(C_cpu); free(C_auto);

    // ---- 性能对比 ----
    printf("\n=== Performance (GFlops) ===\n");
    const char* sep = gpu_ok ? "  |  %-10s %-10s %-10s %-10s\n" : "  %s\n";
    if (gpu_ok)
        printf("  %-8s %-10s %-10s %-10s %-10s\n",
               "Size", "CPU", "Auto", "GPU", "GPU/CPU");
    else
        printf("  %-8s %-10s\n", "Size", "CPU Auto");

    for (int si = 0; si < num_sizes; si++) {
        N = sizes[si];
        double t_cpu = 0, t_auto = 0, t_gpu = 0;

        float *A = (float*)malloc(sizeof(float) * N * N);
        float *B = (float*)malloc(sizeof(float) * N * N);
        float *C = (float*)malloc(sizeof(float) * N * N);
        gen_matrix(A, N, 0.5f);
        gen_matrix(B, N, 0.3f);

        for (int r = 0; r < runs; r++) {
            memset(C, 0, sizeof(float) * N * N);
            Timer t; t.begin();
            matmul_cpu(A, B, C, N, N, N);
            t_cpu += t.ms();
        }
        t_cpu /= runs;

        for (int r = 0; r < runs; r++) {
            memset(C, 0, sizeof(float) * N * N);
            Timer t; t.begin();
            matmul(A, B, C, N, N, N);
            t_auto += t.ms();
        }
        t_auto /= runs;

        if (gpu_ok) {
            for (int r = 0; r < runs; r++) {
                memset(C, 0, sizeof(float) * N * N);
                Timer t; t.begin();
                matmul_gpu(A, B, C, N, N, N);
                t_gpu += t.ms();
            }
            t_gpu /= runs;
            printf("  %-8d %8.1f  %8.1f  %8.1f   %5.1fx\n",
                   N, gflops(N,N,N,t_cpu), gflops(N,N,N,t_auto),
                   gflops(N,N,N,t_gpu),
                   t_cpu/t_gpu);
        } else {
            printf("  %-8d %8.1f\n", N, gflops(N,N,N,t_auto));
        }

        free(A); free(B); free(C);
    }

#ifdef MATMUL_HAS_METAL
    matmul_gpu_release();
#endif

    printf("\nDone.\n");
    return 0;
}
