# Metal GPU 编程学习笔记

> 从 Metal 框架接口到 GPU/CPU 物理差异的完整学习路径，基于实际矩阵乘法优化项目。

---

## 目录

1. [ObjC/C++ 联编：为什么 .mm 是必须的](#1-objcc-联编为什么-mm-是必须的)
2. [Metal 编程模型：谁做什么](#2-metal-编程模型谁做什么)
3. [GPU 线程层级：grid → threadgroup → simdgroup → thread](#3-gpu-线程层级grid--threadgroup--simdgroup--thread)
4. [GPU 内存层级：VRAM → SRAM → Register](#4-gpu-内存层级vram--sram--register)
5. [三个 kernel 的逐级优化](#5-三个-kernel-的逐级优化)
6. [simdgroup：硬件矩阵乘法的抽象边界](#6-simdgroup硬件矩阵乘法的抽象边界)
7. [CPU vs GPU 物理差异 → 优化策略差异](#7-cpu-vs-gpu-物理差异--优化策略差异)
8. [性能数据总结](#8-性能数据总结)

---

## 1. ObjC/C++ 联编：为什么 .mm 是必须的

### 问题
Metal 是 Apple 的 GPU 框架，只提供 ObjC 和 Swift 官方接口，没有 C/C++ API。

### 解决方案：`.mm` 文件（Objective-C++）

```
┌──────────────────────┐     ┌──────────────────────────┐
│ test_metal.cpp       │     │ metal_matmul.mm          │
│ (纯 C++, .cpp)       │────>│ (ObjC++, .mm)            │
│                      │     │                          │
│ include 纯 C 头文件   │     │ #import <Metal/Metal.h>  │
│ 只看到 MetalContext*  │     │ 内部持有 ObjC 对象       │
│ 调用 C 函数          │     │ 封装所有 ObjC 调用       │
└──────────────────────┘     └──────────────────────────┘
```

### 关键模式：不透明指针

```cpp
// metal_matmul.h (纯 C++)
struct MetalContext;  // 前向声明，调用方不知道内部

// metal_matmul.mm (ObjC++)
struct MetalContext {
    id<MTLDevice> device;    // 内部持有 ObjC 对象
    id<MTLCommandQueue> queue;
};
```

调用方只需要 `#include "metal_matmul.h"`，完全不需要知道 ObjC 的存在。

### ObjC 是胶水，Metal Shader 才是核心

```
.metal 文件 (MSL) → 编译器 → GPU 机器码 → MTLComputePipelineState
                                                    ↑
ObjC/Swift 代码只是: 分配内存 → 设置参数 → 提交给 GPU → 等结果
                    不参与任何实际计算！
```

---

## 2. Metal 编程模型：谁做什么

### Metal 对象层次

```
MTLDevice           ← GPU 硬件抽象
  ├─ MTLCommandQueue    ← 命令提交通道
  │   └─ MTLCommandBuffer ← 一组 GPU 命令
  │       └─ MTLComputeCommandEncoder ← 编码计算任务
  │           · setPipelineState()   选 kernel
  │           · setBuffer()          绑参数
  │           · dispatchThreads()    分发线程
  │
  ├─ MTLLibrary        ← 编译后的 shader
  │   └─ MTLComputePipelineState ← GPU 可执行代码
  │
  └─ MTLBuffer         ← GPU 内存
```

### 标准流程

```
1. 创建 MTLBuffer (GPU 内存)
2. 拷贝输入数据到 buffer
3. 创建 CommandBuffer
4. 创建 ComputeCommandEncoder
5. 设置 pipeline state (选 kernel)
6. 绑定参数 (buffer(0), buffer(1), ...)
7. 设置 threadgroup memory (可选)
8. dispatchThreads (指定线程数)
9. endEncoding
10. commit (提交给 GPU!)
11. waitUntilCompleted
12. 读回结果
```

### MTLStorageMode 的选择

- **Shared**：CPU/GPU 共享同一物理内存。在 Apple Silicon 上零拷贝。适用单向数据流场景（写一次 → 读一次）。
- **Private**：GPU 独占。GPU 可以用更激进的缓存策略，但需要 blit copy 搬数据。

**矩阵乘法场景：Shared 是最优选择**，因为 CPU 写完 A、B 就不再碰它们，不存在乒乓访问的 coherence 开销。

---

## 3. GPU 线程层级：grid → threadgroup → simdgroup → thread

```
grid (全部线程)
├── threadgroup (0)           ← 共享 threadgroup memory (SRAM)
│   ├── simdgroup (0)         ← 32 线程锁步执行
│   │   ├── thread 0          ← 持有寄存器
│   │   ├── thread 1
│   │   └── ...
│   ├── simdgroup (1)
│   └── ...
├── threadgroup (1)
└── ...
```

### 与 CPU 概念的对应（不完全精确但有启发）

| GPU 概念 | 近似 CPU 类比 | 关键差异 |
|----------|-------------|---------|
| **simdgroup** (32 threads) | CPU thread（指令流的载体） | simdgroup 的指令作用于 32 个数据 lane |
| **thread** | SIMD lane（数据槽位） | 只是一组寄存器的"所有权"，没有独立指令流 |
| **threadgroup** | 没有好的对应 | 最接近：绑核的一组线程 + 手动管理的 scratchpad |
| **Compute Unit** | CPU 核心 | CU 可管理多个 threadgroup 并在其间切换 |

### 2D 线程索引

GPU 原生支持 3D 线程索引（图形渲染的遗产：纹理坐标、体素等），矩阵乘法自然用 2D：

```metal
uint row = gid.y;  // 矩阵的行
uint col = gid.x;  // 矩阵的列
```

---

## 4. GPU 内存层级：VRAM → SRAM → Register

| 内存类型 | 物理位置 | 大小 | 延迟 | 可见范围 |
|---------|---------|------|------|---------|
| **Register** | GPU 核心内 | 每线程 ~128-256 个 | **0 cycle** | 单线程私有 |
| **Threadgroup SRAM** | GPU 芯片 SRAM | 32KB/threadgroup | ~20 cycles | threadgroup 内共享 |
| **Device (VRAM)** | 统一内存 LPDDR | 几 GB | ~200-400 cycles | 所有线程 |

### 与 CPU 的关键差异

| | CPU L1 Cache | GPU Threadgroup SRAM |
|---|---|---|
| 管理方式 | **硬件自动**换入换出 | **软件显式**加载/同步 |
| 数据单位 | 64B cache line | 4B bank |
| 跳步访问代价 | 浪费 cache line 带宽 (1/16) | ~1 cycle bank conflict |
| 转置优化有效吗 | **有效** — 消除 cache line 浪费 | **效果很小** |

**为什么 SRAM 不在意跳步？**
- CPU cache line 是 64B 粒度：读 1 个 float，搬 16 个 float，跳步时浪费 15/16
- GPU SRAM bank 是 4B 粒度：读 1 个 float，只搬 4 字节，不存在"多搬浪费"

---

## 5. 三个 kernel 的逐级优化

### Stage 1: naive_matmul

每个线程计算 C 的一个元素，直接从 VRAM 读取 A 和 B。

```metal
for (uint k = 0; k < K; k++) {
    sum += A[row * K + k] * B[k * N + col];
}
```

**问题**：A 的每个元素被同一行 N 个线程重复读取，B 被同一列 M 个线程重复读取。

### Stage 2: tiled_matmul

256 线程协作把 A 和 B 的 tile 加载到 SRAM，然后从 SRAM 计算。

```metal
// 协作加载 (256 线程各加载 1 个元素，填满 16×16 tile)
As[tid.y * TILE_SIZE + tid.x] = A[row * K + (k_block + tid.x)];
Bs[tid.y * TILE_SIZE + tid.x] = B[(k_block + tid.y) * N + col];
threadgroup_barrier(...);

// 从 SRAM 计算 (复用!)
for (uint kk = 0; kk < TILE_SIZE; kk++) {
    sum += As[tid.y * TILE_SIZE + kk] * Bs[kk * TILE_SIZE + tid.x];
}
```

**关键洞察**："协作"不是一行代码——是 256 个线程各执行一遍"加载自己的那 1 个元素"，256 个元素恰好填满 tile 的涌现结果。

### Stage 3: simd_matmul

在 tiled 基础上，再次把数据从 SRAM 搬到寄存器，用硬件矩阵乘法单元计算。

```metal
// 从 SRAM 加载到寄存器 (32 线程集体操作)
simdgroup_load(A_sg, As, 8);
simdgroup_load(B_sg, Bs, 8);

// 硬件矩阵乘加 (1 条指令, 512 次 FMA!)
simdgroup_multiply_accumulate(C_sg, A_sg, B_sg, C_sg);
```

**优化链条**：
```
naive:  DRAM → 计算     (每 FMA 都有 DRAM 延迟)
tiled:  DRAM → SRAM → 计算   (DRAM 只读一次, SRAM 20 cycle)
simd:   DRAM → SRAM → Reg → 计算  (最终计算全寄存器, 0 cycle)
```

---

## 6. simdgroup：硬件矩阵乘法的抽象边界

### simdgroup_float8x8 是什么

- 8×8 = 64 个 float 的矩阵，分布在 32 个线程的私有寄存器中
- 每个线程持有 2 个元素
- 对开发者是**黑盒**——你用 `simdgroup_load/store/multiply_accumulate` 操作它

### 单线程视角 vs 集体操作

```metal
// 普通操作：单线程视角
As[t_row * 8 + t_col] = A[...];  // 每个线程独立执行

// 集体操作：simdgroup 级别
simdgroup_load(A_sg, As, 8);     // 32 线程同时，硬件分布数据
simdgroup_multiply_accumulate(C_sg, A_sg, B_sg, C_sg);  // 硬件矩阵单元
threadgroup_barrier(...);        // 所有线程互相等待
```

**关键**：你从头到尾写的都是单线程视角。普通操作被复制到每个线程独立执行（访问不同地址），集体操作也是每个线程执行的同一行代码，但硬件知道"你们是一伙的"。

### simdgroup_multiply_accumulate 内部

**不是一个线程做更多事。是 32 个线程把数据凑在寄存器里，硬件矩阵乘法单元直接跨线程读取全部寄存器，一次性完成 512 次 FMA。**

单个线程只负责"提供并保管 2 个寄存器值"，乘法是矩阵单元干的，和线程没关系。

### 学习到此足够

这是 GPU 编程的抽象边界：

```
你需要知道的：              你不需要知道的：
simdgroup_load              矩阵单元内部布线
simdgroup_mul_acc           寄存器分配算法
simdgroup_store             硬件调度器策略
threadgroup_barrier         锁步执行的微架构
thread_position_in_grid     warp scheduler 轮转
```

---

## 7. CPU vs GPU 物理差异 → 优化策略差异

### 核心架构差异

| | CPU | GPU |
|---|---|---|
| 设计哲学 | 低延迟：少量线程，乱序执行 | 高吞吐：大量线程，隐藏延迟 |
| 线程数 | ~10 量级 | ~10000 量级 |
| 寄存器/核心 | ~32KB | ~256KB per CU |
| Cache 管理 | 硬件自动 | SRAM 由软件显式管理 |
| FMA 来源 | SIMD 指令 (NEON 4-wide, AVX 8-wide) | 专用矩阵乘法单元 |

### 为什么 GPU 的寄存器比 CPU 多这么多？

因果关系：**因为要有大量 in-flight 线程 → 需要大量寄存器存上下文 → 设计时配大寄存器文件。**

```
闭环：
  大寄存器 → 多 in-flight 线程 → 隐藏 DRAM 延迟
  → 需要高带宽喂数据 → 设计高带宽 DRAM
  → 能支持更多线程 → 更需要大寄存器
```

### CPU 优化策略 vs GPU 优化策略

| 优化目标 | CPU 做法 | GPU 做法 |
|---------|---------|---------|
| 数据复用 | 分块 + **转置** (消除 cache line 浪费) | 分块 (SRAM bank 架构不浪费) |
| 数据邻近 | **手动 SIMD** (vmlaq_f32) | **硬件自动** (写标量代码，SIMT 自动向量化) |
| 减少 DRAM 访问 | 分块 + L1/L2 亲和 | 协作加载到 SRAM + simdgroup 到寄存器 |
| 线程同步 | OS + mutex | threadgroup_barrier |
| 缓存管理 | **硬件自动** | **你**手动管理 SRAM 加载 |

### 转置在 CPU 有效、在 GPU tiled 之后没必要

```
CPU:
  分块让数据在 L1 中 → 但 L1 按 64B cache line 管理
  → 按列访问 = 每 cache line 只用 1/16 → 必须转置

GPU:
  分块让数据在 SRAM 中 → SRAM 按 4B bank 管理
  → 按列访问 = bank conflict (~1 cycle)，无带宽浪费
  → 转置不必要
```

### GPU 为什么比 CPU 快

三个因素叠加：

1. **大规模并行**：M2 Pro 有 19 CU，每 CU 可同时管理 ~1024 线程 = ~19K in-flight threads
2. **大量寄存器**：支持这么多线程同时活跃，任一线程等 DRAM 时调度器切换到另一线程
3. **高带宽**：统一内存 ~200 GB/s，且 GPU 的内存请求模式更能喂饱 DRAM 控制器

### 为什么 AMX（CPU 协处理器）比 GPU 快 2 倍？

实测数据（M2 Pro, 2048×2048）：Accelerate/AMX ≈ 2000 GFlops vs GPU simdgroup ≈ 790 GFlops。

直觉上 GPU 有 19 个 CU、"几千个核"，应该更快。但实际相反，原因：

**1. AMX 是专用矩阵引擎，GPU 是通用计算单元**

```
AMX (每个性能核 1 个, 共 8 个):         GPU (19 CU):
┌────────────────────────┐          ┌──────────────────────┐
│ 专用矩阵乘法流水线       │          │ 通用 SIMD lane        │
│ 硅片面积全部给 matmul   │          │ 通过 simdgroup 指令    │
│ 宽度: 远大于 8×8        │          │ "拼出"8×8 矩阵乘法     │
│ 每 cycle: 数百 FMA      │          │ 还要处理:              │
│ 只做一件事: matmul      │          │  · 加载/存储指令       │
│ 不跑 shader 代码        │          │  · barrier 同步        │
│ 没有分支/调度开销       │          │  · 边界检查            │
│                        │          │  · 地址计算            │
└────────────────────────┘          └──────────────────────┘
```

**2. 专用硬件 vs 通用硬件的铁律**

同样面积的硅片，专用电路永远比通用电路快。AMX 把所有晶体管都用来做矩阵乘法——更宽的数据通路、更多的乘法器、更深的流水线。GPU CU 的晶体管要兼顾整数运算、浮点运算、分支预测、线程调度、纹理采样……

就像：专用快递卡车 vs 公交车顺路带货。公交车座位多（GPU 线程多），但货物装卸效率远不如专用卡车。

**3. 我们的 GPU kernel 利用率不够高**

当前 simdgroup kernel 只用 32 线程/threadgroup，GPU occupancy 远未饱和。理论上调优后（多 simdgroup/threadgroup + 双缓冲 + 更大 K tile）可能追到 ~1.5 TFlops，但仍难超过 AMX。

**4. M 系列芯片的三层计算栈（最终版）**

```
层级              用于矩阵乘法             实测 GFlops (2048x2048)
─────────────────────────────────────────────────────────────
NEON SIMD         通用 ARM 向量指令         ~150
                  4-wide FP32 FMA

GPU simdgroup     通用 GPU 计算单元         ~790
                  经 Metal shader 调用

AMX 协处理器      专用矩阵乘法引擎          ~2000
                  Apple 私有, 仅通过
                  Accelerate.framework 调用
```

**结论：如果目标是"最快的矩阵乘法"，直接调 `cblas_sgemm`。自己写 GPU kernel 的收益不在绝对速度，而在理解 GPU 计算模型。**

---

## 8. 性能数据总结

测试环境：Apple M2 Pro，矩阵乘法 C = A × B

### 2048×2048 完整对比

| 实现 | GFlops | 相对 Naive | 关键技术 |
|------|--------|-----------|---------|
| CPU Naive | 0.7 | 1.0x | 三重循环 |
| CPU SIMD+MT (手写 NEON) | 155 | 221x | NEON 4-wide + 8核 |
| GPU Naive | 389 | 556x | 百万线程 |
| GPU Tiled | 684 | 977x | SRAM 分块 |
| GPU Simd | 790 | 1129x | 硬件矩阵单元 |
| **CPU Accelerate (AMX)** | **~2000** | **2857x** | **Apple AMX 协处理器** |

### 优化收益来源

```
CPU Naive → SIMD+MT:    +154 GFlops  来自 NEON SIMD + 多核并行
SIMD+MT → GPU Naive:     +234 GFlops  来自百万线程 + 高带宽
GPU Naive → GPU Tiled:   +295 GFlops  来自 DRAM→SRAM 数据复用
GPU Tiled → GPU Simd:    +106 GFlops  来自 SRAM→寄存器 + 硬件矩阵单元
GPU Simd → AMX:         +1210 GFlops  来自专用矩阵引擎 vs 通用 GPU

最终: AMX 是手写 NEON 的 13 倍，是 GPU simdgroup 的 2.5 倍。
差距不在"优化技巧"，而在硬件架构——专用 > 通用。
```

---

## 关键认知总结

1. **Metal 是接口，MSL 是核心**：ObjC/Swift 只是胶水，实际逻辑在 `.metal` 文件的 kernel 函数里
2. **GPU 写标量，硬件做并行**：你写 `sum += a * b`，硬件在几千线程上同时跑并自动向量化
3. **Threadgroup SRAM 是 GPU 编程的核心优化手段**：相当于 CPU 的 L1 cache，但是手动管理的
4. **simdgroup 是应用层 GPU 开发的终点**：到此为止，再往下是硬件工程师的领域
5. **同一优化在不同架构收益不同**：转置对 CPU 有效（消除 cache line 浪费），对 GPU tiled 之后几乎无效（SRAM bank 无此问题）
6. **GPU 快不是因为 DRAM 更快**：物理芯片一样，但 GPU 的内存访问模式让 DRAM 控制器效率更高
