#include "matrix.h"
#include <iostream>

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

    // 如果有参数，调用test_mod
    if (argc >= 2 && (strcmp(argv[1], "-t") == 0 || strcmp(argv[1], "--test") == 0)) {
        test_mod(argc - 2, argv + 2);
        return 0;
    }

    // 否则运行一个简单的测试
    std::cout << "Running simple matrix multiplication test..." << std::endl;
    
    const int N = 512;
    float *A = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
    float *B = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
    float *C = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * N * N));
    
    if (!A || !B || !C) {
        std::cerr << "Memory allocation failed!" << std::endl;
        return 1;
    }
    
    // 生成测试矩阵
    matrix_gen(A, B, N, 0.3f);
    
    // 测试async_simd方法
    auto start = std::chrono::high_resolution_clock::now();
    float* result = async_matrix_mul_trans_block_with_simd(A, B, C, N, N, N, 64, 4);
    auto end = std::chrono::high_resolution_clock::now();
    
    double elapsed_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    double gflops = (2.0 * N * N * N) / (elapsed_ms * 1e6);
    float trace = Trace(result, N, N);
    
    std::cout << "Matrix size: " << N << "x" << N << std::endl;
    std::cout << "Execution time: " << elapsed_ms << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFlops" << std::endl;
    std::cout << "Trace: " << trace << std::endl;
    
    // 清理内存
    free(A);
    free(B);
    free(C);
    
    std::cout << "\nUse '" << argv[0] << " --test --help' for advanced testing options." << std::endl;
    
    return 0;
}