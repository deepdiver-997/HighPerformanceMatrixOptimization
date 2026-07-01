// =============================================================================
// Metal Shader — GPU 矩阵乘法 kernel
//
// 这个文件包含在 GPU 上运行的代码，用 Metal Shading Language (MSL) 编写。
// MSL 基于 C++14，但有一些关键的区别和扩展。
//
// 【GPUs 编程快速入门（这和 CPU 编程完全不同！）】
//
// 1. SIMD 模型 (不是 SIMD 指令!)
//    CPU: 你写一个函数 → 一个线程执行它
//    GPU: 你写一个 kernel → 成百上千个线程同时执行它
//    GPU 是 "单指令多线程" (SIMT) 模式:
//      - 同一段代码在几千个线程上同时运行
//      - 每个线程处理不同的数据 (通过 thread_position_in_grid 区分)
//
// 2. 线程层级 (从大到小):
//    grid        = 全部线程的总集合 (对应整个输出矩阵)
//    threadgroup = 一组可以协作的线程 (共享 on-chip 内存)
//    simdgroup   = 32 个"锁步"线程 (执行完全相同的指令)  [后续学习]
//    thread      = 单个线程
//
//    类比 CPU 多线程:
//      grid        ≈ 整个并行任务
//      threadgroup ≈ 一个 CPU 核心上的一组线程 (共享 L1 cache)
//      thread      ≈ 单个 CPU 线程
//
// 3. 内存层级 (从快到慢):
//    Register   — 每个线程私有，最快 (TB/s)
//    Threadgroup — on-chip SRAM，threadgroup 内共享 (数 TB/s)
//    Device     — VRAM 或统一内存，所有线程可访问 (数百 GB/s)
//
//    类比 CPU:
//      Register   ≈ CPU 寄存器
//      Threadgroup ≈ L1 cache (但手动管理！)
//      Device     ≈ 主内存 RAM
//
// 4. 关键: GPU 通过"大量线程 + 快速切换"来隐藏内存延迟
//    当一个线程在等待内存时，GPU 切换到另一个线程执行
//    这就是为什么 GPU 需要几千个线程才能达到峰值性能
//
// 【Metal Shading Language vs C++】
//   相同: 基本语法、类型 (float, int, uint)、for/while、指针
//   新增: kernel 函数、地址空间限定符 (device/threadgroup/constant)
//        向量类型 (float4, uint2)、内置函数
//   限制: 无递归、无虚函数、无异常、无 RTTI、部分 C++ 特性不可用
// =============================================================================

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// 阶段1: 朴素 GPU 矩阵乘法 (naive_matmul)
//
// 【设计思路】
//   每个 GPU 线程计算输出矩阵 C 的一个元素
//   C[row][col] = sum(A[row][k] * B[k][col])  for k = 0..K-1
//
// 【线程映射】
//   thread_position_in_grid (简称 gid) 是 Metal 内置的线程标识:
//     gid.x = 线程在 grid 中的 x 坐标 (对应矩阵的列)
//     gid.y = 线程在 grid 中的 y 坐标 (对应矩阵的行)
//
//   例如 M=1024, N=1024，使用 16x16 threadgroup:
//     grid size = (1024, 1024) → GPU 创建 1024x1024 = 1,048,576 个线程
//     线程 (gid.x=5, gid.y=3) 计算 C[3][5]
//
// 【地址空间限定符 (Address Space Qualifiers)】
//   device:    全局内存 (VRAM/统一内存)，所有线程可访问，最慢
//   threadgroup: 片上共享内存，同一 threadgroup 的线程可访问，较快
//   constant:  只读常量内存，所有线程可访问，已优化为快速读取
//   注意: MSL 要求所有指针参数必须有地址空间限定符！
//
// 【性能分析 — 为什么朴素版本慢？】
//   问题1: 内存访问不连续 (strided access)
//     访问 B[k * N + col]: 不同线程的 col 不同，相邻线程读的不是连续地址
//     例如线程 (col=0) 读 B[k*N+0]，线程 (col=1) 读 B[k*N+1] → 还行
//     但访问 A[row * K + k]: 同一行所有线程读相同的 A 元素 → 大量重复读取！
//
//   问题2: 没有利用数据复用
//     A 的每个元素被 N 个线程使用 (同一行的所有线程)
//     B 的每个元素被 M 个线程使用 (同一列的所有线程)
//     但朴素版本每个线程都独立从 VRAM 读取，浪费了复用机会
//
//   问题3: K 维度的循环在全局内存上
//     内层循环每次迭代都访问 A 和 B 的全局内存，延迟很高
// =============================================================================
kernel void naive_matmul(
    // buffer(0): 输入矩阵 A, M x K, row-major
    device const float* A  [[buffer(0)]],
    // buffer(1): 输入矩阵 B, K x N, row-major
    device const float* B  [[buffer(1)]],
    // buffer(2): 输出矩阵 C, M x N, row-major (由 GPU 写入)
    device float*       C  [[buffer(2)]],
    // buffer(3)-(5): 矩阵维度 (constant = 只读常量)
    constant uint& M       [[buffer(3)]],
    constant uint& K       [[buffer(4)]],
    constant uint& N       [[buffer(5)]],
    // thread_position_in_grid: GPU 自动传入，标识当前线程在 grid 中的位置
    // uint2 是 Metal 的二维向量类型
    uint2 gid              [[thread_position_in_grid]])
{
    uint row = gid.y;  // 当前线程负责的输出行
    uint col = gid.x;  // 当前线程负责的输出列

    // 边界检查: 如果矩阵尺寸不是 threadgroup 尺寸的整数倍，
    // 可能会有多余的线程超出矩阵范围
    if (row >= M || col >= N) return;

    float sum = 0.0f;

    // 沿 K 维度遍历，累加 A[row][k] * B[k][col]
    // 这是三重循环的最内层展开到每个线程
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }

    C[row * N + col] = sum;
}

// =============================================================================
// 阶段2: 分块 GPU 矩阵乘法 (tiled_matmul)
//
// 【核心思想: Threadgroup Memory 分块】
//
//   朴素版本的瓶颈: 每个线程从 VRAM 重复读取相同的 A 和 B 元素。
//   例如 M=N=K=1024: A 的每个元素被 1024 个线程读取，B 的每个元素也被
//   1024 个线程读取。总读取量: 2 * 1024 * 1024 * 1024 次 float 读取。
//
//   分块策略:
//     1. 把 A 和 B 分成 16x16 的小块 (tile)
//     2. 一个 threadgroup (16x16=256 个线程) 协作加载一个 tile
//     3. 加载到 threadgroup memory (快！on-chip SRAM)
//     4. 所有线程从 threadgroup memory 计算 (不再反复读 VRAM)
//     5. 沿 K 维度滑动到下一个 tile，重复
//
//   效果: A 的每个元素只从 VRAM 读取 1 次 (而不是 1024 次)
//         B 的每个元素只从 VRAM 读取 1 次 (而不是 1024 次)
//
// 【Threadgroup Memory 详解】
//
//   物理位置: 集成在 GPU 芯片内的 SRAM (和 CPU L1 cache 类似的物理介质)
//   容量:    Apple GPU 通常 32KB/threadgroup (M2 Pro)
//   延迟:    ~20 个时钟周期 (vs device memory ~200-400 cycles)
//   管理方式: 软件显式管理 (不是硬件自动换入换出)
//
//   这既是优势也是负担:
//     优势: 精确控制什么数据在 "cache" 中，避免 cache miss
//     负担: 开发者必须手动加载/同步/边界处理
//
// 【协作加载 (Cooperative Loading)】
//
//   一个 16x16 的 threadgroup 有 256 个线程。
//   每个线程负责加载 A_tile 和 B_tile 的一个元素:
//     线程 (tid.y, tid.x) 加载 As[tid.y][tid.x] 和 Bs[tid.y][tid.x]
//   然后 threadgroup_barrier 同步，确保所有人都加载完成。
//
//   这比自己加载自己需要的数据更快:
//     256 个线程发出 256 个内存请求 (并行)
//     vs 1 个线程发出 256 个请求 (串行)
//
// 【Threadgroup Barrier — GPU 的同步机制】
//
//   threadgroup_barrier(mem_flags::mem_threadgroup):
//     1. 阻塞当前线程，直到 threadgroup 内所有线程都到达这个 barrier
//     2. 确保 barrie 之前的所有 threadgroup memory 写入对 barrie 之后可见
//
//   类比: OpenMP 的 #pragma omp barrier, CUDA 的 __syncthreads()
//
//   为什么需要两个 barrier?
//     barrier 1 (加载后): 确保所有数据加载完成才能开始计算
//     barrier 2 (计算后): 确保所有线程完成计算才能开始加载下一个 tile
//                          (否则可能覆盖还没读完的 tile 数据)
//
// 【内存访问模式分析】
//
//   As[tid.y * TILE_SIZE + tid.x] = A[row * K + (k_block + tid.x)]
//   As 是 16x16, row-major:
//     - 同一行 (同 tid.y) 的不同线程 (不同 tid.x) 访问连续的 A 元素 → coalesced ✓
//     - 每个线程加载一个 float
//
//   Bs[tid.y * TILE_SIZE + tid.x] = B[(k_block + tid.y) * N + col]
//   Bs 是 16x16, row-major:
//     - 同一行 (同 tid.y), 不同 tid.x → 不同 col → 不连续
//       BUT: col 值相近，通常在同一条 cache line 内 → acceptable
//
//   计算时: As[tid.y][kk] * Bs[kk][tid.x]
//     从 As 读: 同一行 (tid.y), 连续的 kk → 连续读取 ✓
//     从 Bs 读: kk 变化 → 不同行 → 跳步读取，但在 SRAM 内 → 快 ✓
// =============================================================================

constant uint TILE_SIZE = 16;  // 必须是编译时常量 (用于数组大小)

kernel void tiled_matmul(
    device const float* A     [[buffer(0)]],
    device const float* B     [[buffer(1)]],
    device float*       C     [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& K          [[buffer(4)]],
    constant uint& N          [[buffer(5)]],
    // 线程在 grid 中的全局位置 (确定算 C 的哪个元素)
    uint2 gid                 [[thread_position_in_grid]],
    // 线程在 threadgroup 中的局部位置 (确定加载 tile 的哪个元素)
    // tid.x, tid.y 都在 [0, TILE_SIZE-1] 范围内
    uint2 tid                 [[thread_position_in_threadgroup]],
    // threadgroup memory: GPU 片上共享内存
    // CPU 侧通过 setThreadgroupMemoryLength:atIndex: 指定大小
    threadgroup float* As     [[threadgroup(0)]],  // 存放 A 的一个 16x16 块
    threadgroup float* Bs     [[threadgroup(1)]])  // 存放 B 的一个 16x16 块
{
    uint row = gid.y;  // 这个线程负责的输出行
    uint col = gid.x;  // 这个线程负责的输出列

    float sum = 0.0f;

    // =====================================================================
    // 沿 K 维度滑动 tile
    //
    // 图示 (M=N=K=64, TILE=16):
    //
    //     A (64x64)                     B (64x64)
    //   ┌────┬────┬────┬────┐        ┌────────────┐
    //   │ T0 │ T1 │ T2 │ T3 │        │            │
    //   ├────┼────┼────┼────┤        │ T0  T1 ... │
    //   │    │    │    │    │        │            │
    //   │    │    │    │    │        ├────────────┤
    //   └────┴────┴────┴────┘        │ T4  T5 ... │
    //                                │            │
    //   沿 K 滑动:                    ├────────────┤
    //   k_block=0: 加载 A 的第0-15列  │            │
    //   k_block=16: 加载 A 的第16-31列└────────────┘
    //   ...                          k_block=0: 加载 B 的第0-15行
    //                                k_block=16: 加载 B 的第16-31行
    //
    //   每次迭代, threadgroup 使用当前 As 和 Bs tile 计算部分和
    // =====================================================================
    for (uint k_block = 0; k_block < K; k_block += TILE_SIZE) {

        // ---- 协作加载 A 的 tile ----
        // 线程 (tid.y, tid.x) 加载 A 的一个元素
        // 加载模式: 线程的行 (tid.y) → A 的行 (row)
        //           线程的列 (tid.x) → A 的列 (k_block + tid.x)
        //
        // row = threadgroup 的起始行 + tid.y
        // 所以同一 threadgroup 的不同行加载 A 的不同行
        // 同一行的不同列加载 A 当前行的连续列 → coalesced read ✓
        if (row < M && (k_block + tid.x) < K) {
            As[tid.y * TILE_SIZE + tid.x] = A[row * K + (k_block + tid.x)];
        } else {
            As[tid.y * TILE_SIZE + tid.x] = 0.0f;  // 越界位置填充0
        }

        // ---- 协作加载 B 的 tile ----
        // 线程 (tid.y, tid.x) 加载 B 的一个元素
        // 加载模式: 线程的行 (tid.y) → B 的行 (k_block + tid.y)
        //           线程的列 (tid.x) → B 的列 (col)
        //
        // col = threadgroup 的起始列 + tid.x
        // 同一行 (同 tid.y) 的不同 tid.x → 不同 col
        // 访问不连续但空间局部性好 (同一条 cache line)
        if (col < N && (k_block + tid.y) < K) {
            Bs[tid.y * TILE_SIZE + tid.x] = B[(k_block + tid.y) * N + col];
        } else {
            Bs[tid.y * TILE_SIZE + tid.x] = 0.0f;
        }

        // ★ barrier 1: 确保所有线程的加载操作都完成
        // 任何一个线程如果没加载完自己的数据，其他线程就不能开始计算
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- 用 threadgroup memory 计算 ----
        // 现在 As 和 Bs 都在快速 SRAM 中
        // 线程 (tid.y, tid.x) 计算:
        //   sum += As[tid.y][*] · Bs[*][tid.x]
        //
        // As[tid.y][kk]  = A 第 row 行, 第 k_block+kk 列
        // Bs[kk][tid.x]  = B 第 k_block+kk 行, 第 col 列
        // → 计算的正是 C[row][col] 在 k_block..k_block+TILE_SIZE 的部分和
        //
        // 注意: 即使 tile 中有 0.0 (越界填充)，乘加结果也是 0，不影响最终结果
        if (row < M && col < N) {
            for (uint kk = 0; kk < TILE_SIZE; kk++) {
                sum += As[tid.y * TILE_SIZE + kk] * Bs[kk * TILE_SIZE + tid.x];
            }
        }

        // ★ barrier 2: 确保所有线程完成计算，才能安全加载下一个 tile
        // 没有这个 barrier: 某些快的线程可能已经开始覆盖 As/Bs，
        //                   而慢的线程还在读旧的 As/Bs 数据
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 写入最终结果
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// =============================================================================
// 阶段3: simdgroup 矩阵乘加 — 硬件矩阵单元加速
//
// 【核心概念对比】
//
//   tiled_matmul (阶段2):               simd_matmul (阶段3):
//   ─────────────────────               ──────────────────
//   数据在 threadgroup SRAM            数据在寄存器 (0 cycle!)
//   每个 thread 做标量 FMA              32 threads 协作做 8×8 矩阵乘法
//   用通用 SIMD 单元                   用专用矩阵乘法硬件单元
//   ~670 GFlops                       目标 ~1.5-3 TFlops
//
// 【simdgroup_float8x8 是什么？】
//
//   一个 8×8 = 64 个 float 的矩阵，分布在 32 个线程的寄存器中。
//   每个线程持有其中 2 个元素。
//
//   关键：它不是"某个线程的矩阵"。它属于整个 simdgroup。
//   就像 tiled 版本中 threadgroup memory 属于整个 threadgroup 一样，
//   simdgroup_float8x8 是 simdgroup 的"寄存器级共享存储"。
//
//   类比：
//     threadgroup memory = 256 线程共享的 SRAM       (~20 cycle)
//     simdgroup_float8x8  =  32 线程共享的寄存器组    ( 0 cycle)
//
// 【为什么比 tiled 快？】
//
//   tiled 版本的计算循环:
//     for (kk = 0; kk < 16; kk++)
//         sum += As[tid.y * 16 + kk] * Bs[kk * 16 + tid.x];
//         ↑ 每次都从 SRAM 读    ↑ 每次都从 SRAM 读
//         ~20 cycle 延迟        ~20 cycle 延迟
//
//   simd 版本的计算:
//     simdgroup_multiply_accumulate(C_sg, A_sg, B_sg);
//     ↑ 一条指令，数据全在寄存器中
//     ↑ 8×8×8 = 512 FMA = 1024 FLOP，可能在几个 cycle 内完成
//     ↑ 使用 GPU 硬件矩阵乘法单元，不是拼凑的 SIMD
//
// 【线程布局】
//
//   threadgroup = 32 threads (1 simdgroup)
//   每个 threadgroup 计算 C 的一个 8×8 块
//
//   32 个线程在 threadgroup 中编号为 tid = 0..31。
//   所有 32 个线程属于同一个 simdgroup，协作持有 A_sg, B_sg, C_sg。
//
// 【执行流程】
//
//   对每个 k_block (步长 8):
//     1. 32 线程协作加载 A[8×8] 到 threadgroup memory As
//     2. 32 线程协作加载 B[8×8] 到 threadgroup memory Bs
//     3. barrier
//     4. simdgroup_load:  从 SRAM 加载到寄存器 (As → A_sg, Bs → B_sg)
//     5. simdgroup_multiply_accumulate: C_sg += A_sg × B_sg
//        ↑ 纯寄存器操作，整个 8×8×8 乘加在硬件矩阵单元中完成
//     6. barrier
//   所有 k_block 处理完后:
//     7. simdgroup_store: 从寄存器写回 C
//
// 【load/store 的数据布局】
//
//   simdgroup_load(data, ptr, stride):
//     从 ptr 开始加载 8×8 矩阵，stride 是行间距 (float 数)
//     例如 As 是 [8][8] row-major: stride = 8
//
//   simdgroup_store(data, ptr, stride):
//     把 8×8 矩阵写入 ptr，stride 是目标矩阵的行间距
//     例如 C 是 [M][N] row-major: stride = N
// =============================================================================

// TILE_SIZE_K: K 维度的分块大小
// 必须是 8 的倍数，因为 simdgroup_multiply_accumulate 一次处理 8 个 K 元素
// 16 或 32 是常见选择
constant uint SIMD_TILE_K = 8;

kernel void simd_matmul(
    device const float* A      [[buffer(0)]],
    device const float* B      [[buffer(1)]],
    device float*       C      [[buffer(2)]],
    constant uint& M           [[buffer(3)]],
    constant uint& K           [[buffer(4)]],
    constant uint& N           [[buffer(5)]],
    // tgid: 当前 threadgroup 在 grid 中的位置 (以 threadgroup 为单位)
    // tgid.x → 第几个 8 列块, tgid.y → 第几个 8 行块
    uint2 tgid                 [[threadgroup_position_in_grid]],
    // tid: 线程在 threadgroup 内的编号 (0..31)
    ushort tid                 [[thread_index_in_threadgroup]],
    // threadgroup memory: 作为 simdgroup_load 的中间缓冲区
    // 每个 8×8 = 64 floats = 256 bytes
    threadgroup float* As      [[threadgroup(0)]],
    threadgroup float* Bs      [[threadgroup(1)]])
{
    // =========================================================================
    // 第1步: 确定这个 threadgroup 负责的 C 子矩阵
    // =========================================================================
    // 每个 threadgroup 处理一个 8×8 的 C 块
    uint C_row = tgid.y * 8;   // C 子矩阵的起始行
    uint C_col = tgid.x * 8;   // C 子矩阵的起始列

    // 边界: 如果整个 8×8 块都超出矩阵范围，直接返回
    if (C_row >= M || C_col >= N) return;

    // =========================================================================
    // 第2步: 初始化 simdgroup 累加器
    // =========================================================================
    // simdgroup_float8x8 是一个 8×8 矩阵，64 个 float 分布在 32 个线程的寄存器中
    // 每个线程持有 2 个元素，对程序员透明（你不需要关心中间分布）
    // "属于 simdgroup" 意味着: 所有 32 个线程必须协作调用 simdgroup_* 函数
    simdgroup_float8x8 C_sg(0.0f);  // 初始化为全零

    // =========================================================================
    // 第3步: 沿 K 维度滑动，每次处理 8 个元素
    // =========================================================================
    //
    // 每次迭代:
    //   A_sg = A[C_row..C_row+7][k..k+7]  (8×8)
    //   B_sg = B[k..k+7][C_col..C_col+7]  (8×8)
    //   C_sg += A_sg × B_sg               (硬件矩阵乘加)
    //
    // 注意: 当 K 很小时(如 256)，每个 8×8×8 的乘加单元跑 32 个 k_block。
    //   C_sg 在这些迭代中累加所有部分积，全程在寄存器中。
    //
    // t_row, t_col: 这个线程在 8×8 tile 中的位置
    // 只依赖于 tid (0..31)，不随 K 迭代变化，所以声明在循环外
    ushort t_row = tid / 8;   // 0..3
    ushort t_col = tid % 8;   // 0..7

    for (uint k = 0; k < K; k += SIMD_TILE_K) {

        // ---- 3a: 协作加载 A 的 8×8 tile 到 threadgroup memory ----
        //
        // 32 线程加载 64 个元素 → 每线程加载 2 个
        // 线程负责的行: t_row 和 t_row+4
        // 线程负责的列: t_col
        //   总共: 4×8 + 4×8 = 64 ✓

        // 全局 A 中的位置: A[C_row + local_row][k + local_col]
        uint a_row0 = C_row + t_row;
        uint a_row1 = C_row + t_row + 4;
        uint a_col  = k + t_col;

        // 加载并处理边界
        As[t_row * 8 + t_col]       = (a_row0 < M && a_col < K) ? A[a_row0 * K + a_col] : 0.0f;
        As[(t_row + 4) * 8 + t_col] = (a_row1 < M && a_col < K) ? A[a_row1 * K + a_col] : 0.0f;

        // ---- 3b: 协作加载 B 的 8×8 tile 到 threadgroup memory ----
        //
        // B 的 8×8 tile: B[k..k+7][C_col..C_col+7]
        // 全局位置: B[k + row][C_col + col]
        //
        uint b_row0 = k + t_row;
        uint b_row1 = k + t_row + 4;
        uint b_col  = C_col + t_col;

        Bs[t_row * 8 + t_col]       = (b_row0 < K && b_col < N) ? B[b_row0 * N + b_col] : 0.0f;
        Bs[(t_row + 4) * 8 + t_col] = (b_row1 < K && b_col < N) ? B[b_row1 * N + b_col] : 0.0f;

        // ---- 3c: barrier — 确保所有线程加载完成 ----
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- 3d: 从 threadgroup memory 加载到 simdgroup 寄存器 ----
        //
        // simdgroup_load(src=As, stride=8):
        //   从 As 开始加载 8×8 矩阵，行间距 = 8 float
        //   As 是 [8][8] row-major，所以 stride = 8
        //   加载后，As 的数据进入 32 个线程的寄存器，变成 A_sg
        //
        simdgroup_float8x8 A_sg;
        simdgroup_float8x8 B_sg;

        simdgroup_load(A_sg, As, 8);
        simdgroup_load(B_sg, Bs, 8);

        // ---- 3e: 硬件矩阵乘加 ★ 核心指令 ★ ----
        //
        // C_sg = C_sg + A_sg × B_sg
        //
        // 三个操作数都是 8×8 矩阵，都在寄存器中。
        // 这是 GPU 硬件矩阵乘法单元的一条指令！
        // 在 M2 Pro 上可能只需要几个 cycle 完成 512 次 FMA。
        //
        // 对比 tiled 版本:
        //   tiled:  256 threads × 16 iter × ~40 SRAM cycles × 2 reads
        //   simd:   1 条硬件指令，纯寄存器
        //
        // compute: C_sg = A_sg × B_sg + C_sg
        // 第4个参数是要累加的值 (自己 = 累加)
        simdgroup_multiply_accumulate(C_sg, A_sg, B_sg, C_sg);

        // ---- 3f: barrier — 确保所有线程完成计算再加载下一个 tile ----
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // =========================================================================
    // 第4步: 将累加器写回 C
    // =========================================================================
    //
    // simdgroup_store(data=C_sg, dst=C + C_row*N + C_col, stride=N):
    //   把 C_sg (8×8 矩阵在 32 个线程的寄存器中) 写入 C 的目标位置
    //   stride = N 因为 C 是 M×N row-major
    //
    // 边界处理: 如果 8×8 块部分超出矩阵，只写有效部分
    //
    if (C_row + 8 <= M && C_col + 8 <= N) {
        // 完整块，直接写
        simdgroup_store(C_sg, C + C_row * N + C_col, N);
    } else {
        // 边缘块: 先存到 threadgroup memory，再按边界写
        // (这是处理非 8 倍数矩阵的必要步骤)
        simdgroup_store(C_sg, As, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (ushort i = t_row; i < 8; i += 4) {
            for (ushort j = 0; j < 8; j++) {
                uint r = C_row + i;
                uint c = C_col + j;
                if (r < M && c < N) {
                    C[r * N + c] = As[i * 8 + j];
                }
            }
        }
    }
}
