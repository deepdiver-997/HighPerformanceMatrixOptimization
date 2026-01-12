#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <cstring>

// 从matrix.cpp中提取必要的函数
void* aligned_alloc_helper(size_t align, size_t size) {
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

// 基础矩阵转置分块乘法
float* matrix_mul_trans_block(const float* A, const float* B, float* res, int r, int k, int c, int bs) {
    const size_t align = 64;
    float *b = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * c * k));
    if (!res)
        res = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    if (!b || !res) {
        std::cerr << "Aligned allocation failed" << std::endl;
        free(b);
        free(res);
        return nullptr;
    }
    memset(res, 0, sizeof(float) * r * c);
    
    // 转置B矩阵
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < k; ++j) {
            b[i * k + j] = B[j * c + i];
        }
    }
    
    // 分块计算
    for (int i = 0; i < r; i += bs) {
        int i_end = std::min(i + bs, r);
        for (int j = 0; j < c; j += bs) {
            int j_end = std::min(j + bs, c);
            for (int p = 0; p < k; p += bs) {
                int p_end = std::min(p + bs, k);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = 0.0f;
                        for (int pp = p; pp < p_end; ++pp) {
                            sum += A[ii * k + pp] * b[jj * k + pp];
                        }
                        res[ii * c + jj] += sum;
                    }
                }
            }
        }
    }
    
    free(b);
    return res;
}

// SIMD操作抽象（简化版本）
namespace simd_ops {
    struct simd_f32 {
        float data[4];
    };
    
    inline simd_f32 load(const float* ptr) {
        simd_f32 result;
        for (int i = 0; i < 4; ++i) result.data[i] = ptr[i];
        return result;
    }
    
    inline simd_f32 set1(float value) {
        simd_f32 result;
        for (int i = 0; i < 4; ++i) result.data[i] = value;
        return result;
    }
    
    inline simd_f32 mul(simd_f32 a, simd_f32 b) {
        simd_f32 result;
        for (int i = 0; i < 4; ++i) result.data[i] = a.data[i] * b.data[i];
        return result;
    }
    
    inline simd_f32 add(simd_f32 a, simd_f32 b) {
        simd_f32 result;
        for (int i = 0; i < 4; ++i) result.data[i] = a.data[i] + b.data[i];
        return result;
    }
    
    inline float horizontal_sum(simd_f32 v) {
        float sum = 0.0f;
        for (int i = 0; i < 4; ++i) sum += v.data[i];
        return sum;
    }
}

// SIMD优化的矩阵转置分块乘法
float* matrix_mul_trans_block_with_simd(const float* A, const float* B, float* res, int r, int k, int c, int bs) {
    const size_t align = 64;
    float *b = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * c * k));
    if (!res)
        res = static_cast<float*>(aligned_alloc_helper(align, sizeof(float) * r * c));
    if (!b || !res) {
        std::cerr << "Aligned allocation failed" << std::endl;
        free(b);
        free(res);
        return nullptr;
    }
    memset(res, 0, sizeof(float) * r * c);
    
    // 转置B矩阵
    for (int i = 0; i < c; ++i) {
        for (int j = 0; j < k; ++j) {
            b[i * k + j] = B[j * c + i];
        }
    }
    
    // SIMD分块计算
    for (int i = 0; i < r; i += bs) {
        int i_end = std::min(i + bs, r);
        for (int j = 0; j < c; j += bs) {
            int j_end = std::min(j + bs, c);
            for (int p = 0; p < k; p += bs) {
                int p_end = std::min(p + bs, k);
                
                for (int ii = i; ii < i_end; ++ii) {
                    for (int jj = j; jj < j_end; ++jj) {
                        float sum = 0.0f;
                        int pp_base = p;
                        
                        // 使用SIMD处理4个元素为一组
                        int simd_end = p_end - 3;
                        for (int pp = pp_base; pp <= simd_end; pp += 4) {
                            simd_ops::simd_f32 a_vec = simd_ops::load(&A[ii * k + pp]);
                            simd_ops::simd_f32 b_vec = simd_ops::load(&b[jj * k + pp]);
                            simd_ops::simd_f32 prod = simd_ops::mul(a_vec, b_vec);
                            sum += simd_ops::horizontal_sum(prod);
                        }
                        
                        // 处理剩余元素
                        for (int pp = simd_end + 1; pp < p_end; ++pp) {
                            sum += A[ii * k + pp] * b[jj * k + pp];
                        }
                        
                        res[ii * c + jj] += sum;
                    }
                }
            }
        }
    }
    
    free(b);
    return res;
}

// 测试函数
void test_block_performance(int matrix_size, const std::vector<int>& block_sizes, int test_times = 3) {
    std::cout << "测试矩阵大小: " << matrix_size << "x" << matrix_size << std::endl;
    std::cout << std::setw(8) << "分块大小" << std::setw(15) << "基础版本(ms)" << std::setw(15) << "SIMD版本(ms)" << std::setw(10) << "加速比" << std::endl;
    std::cout << std::string(48, '-') << std::endl;
    
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
    
    std::vector<std::tuple<int, double, double, double>> results;
    
    for (int block_size : block_sizes) {
        // 测试基础版本
        std::vector<double> base_times;
        for (int t = 0; t < test_times; ++t) {
            auto start = std::chrono::high_resolution_clock::now();
            float* C = matrix_mul_trans_block(A, B, nullptr, matrix_size, matrix_size, matrix_size, block_size);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            base_times.push_back(duration.count() / 1000.0);
            free(C);
        }
        
        double base_avg = 0;
        for (double time : base_times) base_avg += time;
        base_avg /= test_times;
        
        // 测试SIMD版本
        std::vector<double> simd_times;
        for (int t = 0; t < test_times; ++t) {
            auto start = std::chrono::high_resolution_clock::now();
            float* C = matrix_mul_trans_block_with_simd(A, B, nullptr, matrix_size, matrix_size, matrix_size, block_size);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            simd_times.push_back(duration.count() / 1000.0);
            free(C);
        }
        
        double simd_avg = 0;
        for (double time : simd_times) simd_avg += time;
        simd_avg /= test_times;
        
        double speedup = base_avg / simd_avg;
        
        std::cout << std::setw(8) << block_size 
                  << std::setw(15) << std::fixed << std::setprecision(2) << base_avg
                  << std::setw(15) << std::fixed << std::setprecision(2) << simd_avg
                  << std::setw(10) << std::fixed << std::setprecision(2) << speedup << std::endl;
        
        results.push_back(std::make_tuple(block_size, base_avg, simd_avg, speedup));
    }
    
    // 找出最优分块大小
    auto best_base = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return std::get<1>(a) < std::get<1>(b); });
    
    auto best_simd = std::min_element(results.begin(), results.end(),
        [](const auto& a, const auto& b) { return std::get<2>(a) < std::get<2>(b); });
    
    std::cout << "\n基础版本最优分块大小: " << std::get<0>(*best_base) 
              << " (时间: " << std::get<1>(*best_base) << " ms)" << std::endl;
    std::cout << "SIMD版本最优分块大小: " << std::get<0>(*best_simd) 
              << " (时间: " << std::get<2>(*best_simd) << " ms)" << std::endl;
    std::cout << "最大加速比: " << std::get<3>(*best_simd) << "x" << std::endl;
    
    // 清理内存
    free(A);
    free(B);
}

int main() {
    std::cout << "=== 矩阵分块性能对比测试 ===" << std::endl;
    std::cout << "测试分块大小: 64, 128, 256, 512" << std::endl;
    
    std::vector<int> block_sizes = {64, 128, 256, 512};
    std::vector<int> matrix_sizes = {512, 1024, 2048};
    
    for (int size : matrix_sizes) {
        std::cout << "\n" << std::string(60, '=') << std::endl;
        test_block_performance(size, block_sizes, 3);
    }
    
    return 0;
}