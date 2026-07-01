// =============================================================================
// matmul.h — 矩阵乘法统一入口
//
// 用法: 只需 #include "matmul.h"，然后调用 matmul()。
//       编译期自动选择最优实现，无需手动选平台。
//
//   #include "matmul.h"
//
//   float *A = ..., *B = ..., *C = ...;
//   matmul(A, B, C, M, K, N);           // 自动选择最优
//   matmul_cpu(A, B, C, M, K, N);       // 强制 CPU
//   matmul_gpu(A, B, C, M, K, N);       // 强制 GPU (如果可用)
//   bool ok = matmul_gpu_available();   // 运行时检测 GPU
//
// 【调度策略 (编译期确定)】
//
//   matmul() 的优先级:
//   ┌─────────────────────────────────────────────────────────┐
//   │ 1. 小矩阵 (M*K*N < SMALL_THRESHOLD)                      │
//   │    → matmul_cpu_small()  直接 CPU 算，避免 GPU 启动开销  │
//   │                                                         │
//   │ 2. MATMUL_HAS_METAL 定义 && 矩阵足够大                    │
//   │    → matmul_gpu()  Metal GPU 硬件矩阵乘法                 │
//   │                                                         │
//   │ 3. MATMUL_HAS_ACCELERATE 定义                             │
//   │    → cblas_sgemm()  Apple AMX 协处理器                    │
//   │                                                         │
//   │ 4. MATMUL_ARCH_ARM_NEON 定义                              │
//   │    → matmul_cpu_neon()  NEON SIMD + 多线程               │
//   │                                                         │
//   │ 5. MATMUL_ARCH_X86_AVX 定义                               │
//   │    → matmul_cpu_avx()  AVX 256-bit + 多线程              │
//   │                                                         │
//   │ 6. MATMUL_ARCH_X86_SSE 定义                               │
//   │    → matmul_cpu_sse()  SSE 128-bit + 分块                │
//   │                                                         │
//   │ 7. 回退                                                   │
//   │    → matmul_cpu_naive()  分块 + 转置，纯标量              │
//   └─────────────────────────────────────────────────────────┘
//
// 【条件编译宏说明】
//
//   以下宏来自 matmul_arch.h，全部是编译期常量:
//
//   平台检测:
//     MATMUL_OS_APPLE         当前是 Apple 系统
//     MATMUL_HAS_METAL        Metal.framework 可用 (需链接)
//     MATMUL_HAS_ACCELERATE    Accelerate.framework 可用 (需链接)
//
//   指令集:
//     MATMUL_ARCH_ARM_NEON    ARM NEON 可用
//     MATMUL_ARCH_X86_AVX     x86 AVX 可用 (256-bit)
//     MATMUL_ARCH_X86_SSE     x86 SSE 可用 (128-bit)
//     MATMUL_ARCH_GENERIC     无 SIMD，纯标量
//
// 【链接要求】
//
//   编译时需要链接:
//     Apple + Metal:  -framework Metal -framework Foundation
//                     同时编译 metal/metal_matmul.mm
//     Apple + Accel:  -framework Accelerate
//     通用:           -pthread
//
//   详见 Makefile
// =============================================================================

#pragma once

#include "matmul_arch.h"  // MATMUL_HAS_METAL, MATMUL_HAS_ACCELERATE, 等
#include <cstddef>        // size_t

// =============================================================================
// 核心 API
// =============================================================================

/**
 * 矩阵乘法 C = A × B  (自动选择最优实现)
 *
 * @param A  输入矩阵, M × K, row-major
 * @param B  输入矩阵, K × N, row-major
 * @param C  输出矩阵, M × N, row-major (调用者分配)
 * @param M  A 的行数
 * @param K  A 的列数 / B 的行数
 * @param N  B 的列数
 */
void matmul(const float* A, const float* B, float* C, int M, int K, int N);

/**
 * CPU 路径 (跳过 GPU，直接走 CPU 最优实现)
 * 内部会选 Accelerate > NEON > AVX > SSE > scalar
 */
void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N);

// =============================================================================
// GPU 路径 (仅 MATMUL_HAS_METAL 时可用)
// =============================================================================

#ifdef MATMUL_HAS_METAL

/**
 * 运行时检测: GPU 是否可用
 * 如果初始化 Metal 失败（比如 shader 文件找不到），返回 false
 */
bool matmul_gpu_available();

/**
 * 初始化 GPU 后端 (必须在 matmul_gpu() 之前调用一次)
 * @param metallib_path  .metal shader 文件路径
 * @return true 成功, false 失败
 */
bool matmul_gpu_init(const char* metallib_path);

/**
 * GPU 矩阵乘法 (使用 Metal simdgroup kernel)
 */
void matmul_gpu(const float* A, const float* B, float* C, int M, int K, int N);

/**
 * 释放 GPU 资源
 */
void matmul_gpu_release();

#endif // MATMUL_HAS_METAL
