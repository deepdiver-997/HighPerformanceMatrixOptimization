#include "matrix.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>

// 测试不同分块大小的性能
void test_block_sizes(int matrix_size, const std::vector<int>& block_sizes, int test_times = 3) {
    std::cout << "测试矩阵大小: " << matrix_size << "x" << matrix_size << std::endl;
    std::cout << std::setw(8) << "分块大小" << std::setw(12) << "平均时间(ms)" << std::setw(12) << "最小时间(ms)" << std::setw(12) << "最大时间(ms)" << std::endl;
    std::cout << std::string(44, '-') << std::endl;
    
    // 创建测试矩阵
    float* A = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * matrix_size * matrix_size));
    float* B = static_cast<float*>(aligned_alloc_helper(64, sizeof(float) * matrix_size * matrix_size));
    
    // 初始化随机矩阵
    srand(42);
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    srand(123);
    for (int i = 0; i < matrix_size * matrix_size; ++i) {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    float* C = nullptr;
    
    std::vector<std::pair<int, double>> results;
    
    for (int block_size : block_sizes) {
        std::vector<double> times;
        
        for (int t = 0; t < test_times; ++t) {
            auto start = std::chrono::high_resolution_clock::now();
            C = matrix_mul_trans_block(A, B, C, matrix_size, matrix_size, matrix_size, block_size);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count() / 1000.0); // 转换为毫秒
        }
        
        double avg_time = 0;
        double min_time = times[0];
        double max_time = times[0];
        
        for (double time : times) {
            avg_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        avg_time /= test_times;
        
        std::cout << std::setw(8) << block_size 
                  << std::setw(12) << std::fixed << std::setprecision(2) << avg_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << min_time
                  << std::setw(12) << std::fixed << std::setprecision(2) << max_time << std::endl;
        
        results.push_back({block_size, avg_time});
    }
    
    // 找出最优分块大小
    auto best = std::min_element(results.begin(), results.end(),
        [](const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.second < b.second; });
    
    std::cout << "\n最优分块大小: " << best->first << " (平均时间: " << best->second << " ms)" << std::endl;
    
    // 使用最优分块大小测试SIMD版本
    std::cout << "\n使用最优分块大小测试SIMD版本..." << std::endl;
    
    std::vector<double> simd_times;
    for (int t = 0; t < test_times; ++t) {
        auto start = std::chrono::high_resolution_clock::now();
        C = matrix_mul_trans_block_with_simd(A, B, C, matrix_size, matrix_size, matrix_size, best->first);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        simd_times.push_back(duration.count() / 1000.0);
    }
    
    double simd_avg = 0, simd_min = simd_times[0], simd_max = simd_times[0];
    for (double time : simd_times) {
        simd_avg += time;
        simd_min = std::min(simd_min, time);
        simd_max = std::max(simd_max, time);
    }
    simd_avg /= test_times;
    
    std::cout << "SIMD版本性能:" << std::endl;
    std::cout << "  平均时间: " << std::fixed << std::setprecision(2) << simd_avg << " ms" << std::endl;
    std::cout << "  最小时间: " << std::fixed << std::setprecision(2) << simd_min << " ms" << std::endl;
    std::cout << "  最大时间: " << std::fixed << std::setprecision(2) << simd_max << " ms" << std::endl;
    
    double speedup = best->second / simd_avg;
    std::cout << "  加速比: " << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;
    
    // 清理内存
    free(A);
    free(B);
    free(C);
}

int main() {
    std::cout << "=== 矩阵分块性能测试 ===" << std::endl;
    
    // 测试不同矩阵大小
    std::vector<int> matrix_sizes = {512, 1024, 2048};
    std::vector<int> block_sizes = {64, 128, 256, 512};
    
    for (int size : matrix_sizes) {
        std::cout << "\n" << std::string(50, '=') << std::endl;
        test_block_sizes(size, block_sizes, 3);
    }
    
    return 0;
}