// =============================================================================
// Metal GPU 矩阵乘法 — 纯 C++ 接口头文件
//
// 【设计思路: 不透明指针 (Opaque Pointer / Pimpl 模式)】
//
//   这个头文件可以被纯 C++ 代码 (.cpp) 安全地 #include。
//   struct MetalContext 只有前向声明，没有定义 — 调用方看不到 ObjC 类型。
//
//   真正的定义在 metal_matmul.mm 中:
//     struct MetalContext {
//         id<MTLDevice> device;   // ObjC 对象
//         id<MTLCommandQueue> queue;
//         ...
//     };
//
//   这样做的优势:
//     1. test_metal.cpp (纯 C++) 不需要 #import <Metal/Metal.h>
//     2. 不需要用 .mm 后缀编译所有文件
//     3. 如果以后移植到其他 GPU API (Vulkan, CUDA)，只需要改 .mm 实现
//
// 【ObjC/C++ 分离的关键】
//
//   ┌──────────────────────┐     ┌──────────────────────────┐
//   │ test_metal.cpp       │     │ metal_matmul.mm          │
//   │ (纯 C++, .cpp)       │────>│ (ObjC++, .mm)            │
//   │                      │     │                          │
//   │ 只 include 这个 .h   │     │ #import <Metal/Metal.h>  │
//   │ 只看到 MetalContext* │     │ #import <Foundation/...> │
//   │ 调用 C 函数接口      │     │ 内部持有 ObjC 对象      │
//   └──────────────────────┘     └──────────────────────────┘
//
//   这就是 ".mm 桥接" 的本质: ObjC 代码被封装在 .mm 文件内，
//   对外只暴露纯 C/C++ 接口。
// =============================================================================

#pragma once

#include <cstddef>   // size_t

// ---- 不透明类型 ----
// 前向声明，定义在 metal_matmul.mm 中
// 调用方只能持有指针，无法直接访问内部成员
struct MetalContext;

// =============================================================================
// 生命周期管理
// =============================================================================

/**
 * 初始化 Metal GPU 上下文
 *
 * 这个函数执行 Metal 的完整初始化流程:
 *   1. 获取系统默认 GPU (MTLCreateSystemDefaultDevice)
 *   2. 加载并编译 .metal shader 源码
 *   3. 从 shader 中提取 kernel 函数，创建 pipeline state
 *   4. 创建命令队列
 *
 * @param metal_source_path  Metal shader 源文件路径 (如 "metal_shaders.metal")
 *                           注意是运行时编译，不是预编译的 .metallib
 * @return MetalContext*     成功返回上下文指针，失败返回 nullptr
 */
MetalContext* metal_init(const char* metal_source_path);

/**
 * 释放 Metal GPU 上下文
 *
 * 析构所有 Metal 对象 (device, queue, library, pipelines)。
 * 由于 ARC，ObjC 对象会自动 release。
 *
 * @param ctx  要释放的上下文 (传 nullptr 是安全的)
 */
void metal_release(MetalContext* ctx);

// =============================================================================
// GPU 矩阵乘法 — 核心计算接口
// =============================================================================

/**
 * 朴素 GPU 矩阵乘法
 *
 * 每个 GPU 线程计算 C 的一个元素，沿 K 维度循环累加。
 * 对应 shader 中的 naive_matmul kernel。
 *
 * 特点:
 *   - 简单直观，但内存效率低
 *   - A 和 B 的全部访问都经过全局内存 (VRAM)
 *   - 适合小矩阵或作为基准对比
 *
 * @param ctx   Metal 上下文
 * @param A     输入矩阵 A (CPU 内存, row-major, M x K)
 * @param B     输入矩阵 B (CPU 内存, row-major, K x N)
 * @param C     输出矩阵 C (CPU 内存, 由调用者分配, M x N)
 * @param M     A 的行数
 * @param K     A 的列数 / B 的行数
 * @param N     B 的列数
 * @return      0 成功, -1 失败
 */
int metal_matmul_naive(MetalContext* ctx,
                       const float* A, const float* B, float* C,
                       int M, int K, int N);

/**
 * 分块 GPU 矩阵乘法 (推荐)
 *
 * 使用 threadgroup 共享内存分块优化。
 * 对应 shader 中的 tiled_matmul kernel。
 *
 * 优化原理:
 *   - 16x16 的 threadgroup 协作加载 A 和 B 的 tile
 *   - tile 存储在 on-chip SRAM (threadgroup memory) 中
 *   - 大幅减少全局内存访问次数 (每个元素只读一次)
 *
 * 性能: 相比 naive 版本通常有 1.5-2x 加速
 *
 * @param ctx   Metal 上下文
 * @param A     输入矩阵 A (CPU 内存, row-major, M x K)
 * @param B     输入矩阵 B (CPU 内存, row-major, K x N)
 * @param C     输出矩阵 C (CPU 内存, 由调用者分配, M x N)
 * @param M     A 的行数
 * @param K     A 的列数 / B 的行数
 * @param N     B 的列数
 * @return      0 成功, -1 失败
 */
int metal_matmul_tiled(MetalContext* ctx,
                       const float* A, const float* B, float* C,
                       int M, int K, int N);

/**
 * simdgroup 矩阵乘法 (最高性能)
 *
 * 使用 GPU 硬件矩阵乘法单元 + simdgroup 协作计算。
 * 对应 shader 中的 simd_matmul kernel。
 *
 * 与 tiled 的关键区别:
 *   - tiled: 数据在 threadgroup SRAM, 每个 thread 独立做标量 FMA
 *   - simd:  数据在寄存器, 32 threads 协作做 8x8 矩阵乘法
 *            simdgroup_multiply_accumulate 映射到 GPU 硬件矩阵单元
 *
 * threadgroup 大小: 32 threads (1 simdgroup)
 * 每个 threadgroup 计算 C 的一个 8x8 块
 *
 * 性能: 目标比 tiled 版本快 2-5x，取决于矩阵大小
 *
 * @param ctx   Metal 上下文
 * @param A     输入矩阵 A (CPU 内存, row-major, M x K)
 * @param B     输入矩阵 B (CPU 内存, row-major, K x N)
 * @param C     输出矩阵 C (CPU 内存, 由调用者分配, M x N)
 * @param M     A 的行数
 * @param K     A 的列数 / B 的行数
 * @param N     B 的列数
 * @return      0 成功, -1 失败
 */
int metal_matmul_simd(MetalContext* ctx,
                      const float* A, const float* B, float* C,
                      int M, int K, int N);

// =============================================================================
// 工具函数
// =============================================================================

/**
 * 获取 GPU 名称
 *
 * @param ctx  Metal 上下文
 * @return     GPU 名称 (如 "Apple M2 Pro")，ctx 无效返回 "N/A"
 */
const char* metal_gpu_name(MetalContext* ctx);

/**
 * 查询 GPU 的 threadgroup 能力
 *
 * @param ctx                          Metal 上下文
 * @param max_threads_per_threadgroup  输出: 每个 threadgroup 的最大线程数
 * @param threadgroup_memory_size      输出: threadgroup memory 最大字节数
 */
void metal_threadgroup_info(MetalContext* ctx,
                            int* max_threads_per_threadgroup,
                            int* threadgroup_memory_size);
