// =============================================================================
// Metal GPU 矩阵乘法 — Objective-C++ 实现
//
// 【文件后缀 .mm 的含义】
//   .mm = Objective-C++  这是关键！
//   它告诉 clang 编译器：这个文件里可以同时写 C++ 和 Objective-C 代码。
//   C++ 代码 (class, template, new/delete, STL) 和 ObjC 代码 (id, @, [] 消息) 可以
//   混合在同一个文件中，编译器会正确处理两者的交互。
//
// 【为什么 Metal 需要 ObjC？】
//   Metal 是 Apple 的 GPU 编程框架。Apple 只提供了两个语言的官方 Metal API：
//     - Objective-C (通过 Metal.framework)
//     - Swift (通过 Metal 模块)
//   没有官方的 C 或 C++ API！
//   虽然有个第三方项目 metal-cpp (Apple 维护的 C++ wrapper)，但它只是封装了 ObjC 调用。
//
//   所以在 .mm 文件里写 Metal 代码是最直接的方式：
//     .mm 文件 → 同时使用 C++ 和 ObjC → 直接调用 Metal API
//
// 【ObjC 和 C++ 如何在这个文件中协作？】
//   ┌─────────────────────────────────────────────────────┐
//   │  metal_matmul.mm  (Objective-C++)                    │
//   │                                                      │
//   │  C++ 部分:                    ObjC 部分:             │
//   │  - struct MetalContext        - id<MTLDevice>        │
//   │  - new / delete               - @autoreleasepool     │
//   │  - fprintf, memcpy, printf    - [obj message]        │
//   │  - static 函数                - @"string literal"    │
//   │  - const char*                - NSString*            │
//   │  - cstdio, cstring            - Metal/Metal.h        │
//   │                                                      │
//   │  关键桥接:                      关键桥接:               │
//   │  NSString* → const char*      const char* → NSString*│
//   │  通过 UTF8String              通过 stringWithUTF8Str│
//   └─────────────────────────────────────────────────────┘
//
// 【ObjC 速览（写给 C++ 开发者）】
//
//   C++                          Objective-C                说明
//   ───                          ───────────                ────
//   class Foo { };               @interface Foo : NSObject  类定义 (头文件)
//                                @end
//
//   Foo* obj = new Foo();        Foo* obj = [[Foo alloc] init];  创建对象
//   delete obj;                  [obj release];  (ARC 自动管理)  释放对象
//
//   obj->method(arg);            [obj method:arg];           调用方法 (消息传递)
//
//   namespace ns { }             无命名空间，用前缀: MTL*, NS*  命名约定
//
//   virtual void foo() = 0;      @protocol Foo               接口/协议
//                                - (void)foo;
//                                @end
//
//   nullptr                       nil                         空指针 (nil 可以安全调用方法)
//
//   std::string                  NSString*                   字符串
//   "hello"                      @"hello"                    ObjC 字符串字面量
//
//   int x;                       NSInteger x;                整数 (架构自适应)
//
//   关键区别:
//   1. ObjC 是动态派发: [obj method] 在运行时查找方法，不是编译时虚表
//   2. ObjC 用消息传递: 向对象发送消息，对象可以响应也可以不响应
//   3. ObjC 的 id 类型: 类似 void* 但可以调用任何方法(编译时不检查类型)
//   4. id<协议> 语法:   id<MTLDevice> 表示"遵循 MTLDevice 协议的任意对象"
//   5. ARC 自动管理内存: 不需要手动 release，编译器插入 retain/release
//   6. @autoreleasepool:  创建自动释放池，类似 C++ 的 RAII 作用域
//      离开 @autoreleasepool 块时，池内所有 ObjC 临时对象被释放
//
// 【GPU 编程关键概念】
//   详见 metal_shaders.metal 的注释。这里从 CPU 侧的视角：
//   - Shader (.metal 文件) 定义 GPU 上运行的 kernel 函数
//   - CPU 通过 Metal API 把数据传给 GPU，告诉 GPU 执行 kernel
//   - Apple Silicon 上 CPU 和 GPU 共享物理内存 (统一内存架构)
//     所以 "拷贝数据到 GPU" 实际上可能不需要物理拷贝
// =============================================================================

// ---- ObjC 头文件 ----
// #import 是 ObjC 版的 #include，自动防止重复包含
// <Metal/Metal.h> 提供所有 GPU API
// <Foundation/Foundation.h> 提供 NSString, NSObject 等基础类型
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// ---- C++ 头文件 ----
// 在 .mm 中可以正常 #include C++ 头文件
#include "metal_matmul.h"  // 我们的纯 C++ 接口
#include <cstdio>
#include <cstring>

// =============================================================================
// MetalContext: 持有 Metal 对象的 C++ struct
//
// 用 C++ struct 封装 ObjC 对象，使得调用方 (test_metal.cpp) 不需要引入 ObjC 头文件
// 这是一个经典的模式: "pimpl with ObjC" 或 "opaque pointer"
//
// 在头文件 metal_matmul.h 中，MetalContext 只是一个前向声明:
//   struct MetalContext;    // 纯 C++ 代码中看不到 ObjC 类型
//
// 在这个 .mm 文件中，我们定义完整的 struct，内部持有 ObjC 对象。
// 这是 ObjC/C++ 桥接的核心技巧。
// =============================================================================
struct MetalContext {
    // id<T> 语法: "遵循协议 T 的对象"
    //   类比 C++: shared_ptr<IMTLDevice>  但更灵活 (运行时类型)
    //   id 本身是万能指针 (类似 void*)，<MTLDevice> 声明它遵循 MTLDevice 协议

    id<MTLDevice>              device;    // GPU 设备对象
    id<MTLCommandQueue>        queue;     // 命令队列：CPU 向 GPU 提交工作
    id<MTLLibrary>             library;   // 编译后的 shader 库

    // MTLComputePipelineState: 编译好的 GPU kernel，可以直接分发执行
    id<MTLComputePipelineState> naive_pipeline;  // 朴素 kernel
    id<MTLComputePipelineState> tiled_pipeline;  // 分块 kernel
    id<MTLComputePipelineState> simd_pipeline;   // simdgroup kernel

    int max_threads_per_threadgroup;
    int threadgroup_memory_size;
};

// =============================================================================
// load_metal_source: 从磁盘读取 .metal 文件内容
//
// 使用纯 C 的 fopen/fread (C++ 部分)
// 结果转换为 NSString* (ObjC 部分)
//
// C 和 ObjC 之间的字符串转换:
//   C → ObjC:  [NSString stringWithUTF8String: c_string]
//   ObjC → C:  [ns_string UTF8String]
// 这两个桥接函数在这个文件里频繁出现
// =============================================================================
static NSString* load_metal_source(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "[Metal] Cannot open shader file: %s\n", path);
        return nil;  // nil 是 ObjC 的空指针, 类似 nullptr
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    rewind(f);

    // C++ 的 new[] 分配缓冲区
    char* buf = new char[sz + 1];
    fread(buf, 1, sz, f);
    buf[sz] = '\0';
    fclose(f);

    // ★ 关键桥接: C 字符串 → ObjC NSString
    NSString* src = [NSString stringWithUTF8String:buf];
    delete[] buf;  // C++ 释放
    return src;    // 返回 ObjC 对象 (ARC 管理生命周期)
}

// =============================================================================
// create_pipeline: 从编译好的 shader 库中取出 kernel 函数，创建 pipeline state
//
// pipeline state = 编译优化后的 GPU 可执行代码 + 参数绑定信息
// 类比: "函数指针" + "签名信息"，可以直接被 GPU 硬件执行
// =============================================================================
static id<MTLComputePipelineState> create_pipeline(id<MTLDevice> device,
                                                    id<MTLLibrary> library,
                                                    NSString* function_name) {
    NSError* err = nil;

    // [obj method:param] 是 ObjC 的方法调用语法 (消息传递)
    // newFunctionWithName: 在 library 中按名称查找 kernel 函数
    // @"naive_matmul" 语法: ObjC 字符串字面量 (注意 @ 前缀)
    id<MTLFunction> func = [library newFunctionWithName:function_name];
    if (!func) {
        // ★ 桥接: ObjC NSString → C 字符串 (用于 fprintf)
        fprintf(stderr, "[Metal] Kernel function '%s' not found in shader\n",
                [function_name UTF8String]);
        return nil;
    }

    // 从 MTLFunction 创建 MTLComputePipelineState
    // 这一步会针对当前 GPU 做最终编译优化
    // error:&err 是 ObjC 的错误传递惯例 (传 NSError** 指针)
    id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:func
                                                                         error:&err];
    if (!pso) {
        fprintf(stderr, "[Metal] Pipeline creation failed: %s\n",
                [[err localizedDescription] UTF8String]);
        return nil;
    }

    // NSLog: ObjC 的 printf，%@ 是 ObjC 对象的格式化占位符 (类似 Java 的 toString)
    NSLog(@"[Metal] Pipeline '%@' created. "
          "Max threads/threadgroup: %lu, "
          "Threadgroup memory: %lu bytes",
          function_name,
          (unsigned long)pso.maxTotalThreadsPerThreadgroup,
          (unsigned long)pso.threadExecutionWidth);

    return pso;
}

// =============================================================================
// metal_init: 初始化 Metal 上下文
//
// 这是整个 Metal 工作的启动点。按顺序:
//   1. 获取 GPU 设备
//   2. 编译 Metal shader 源码
//   3. 创建 pipeline state
//   4. 创建命令队列
//   5. 组装 context
// =============================================================================
MetalContext* metal_init(const char* metal_source_path) {
    // ===== @autoreleasepool 是什么？ =====
    // ObjC 使用引用计数管理内存。在 ARC (Automatic Reference Counting) 下，
    // 编译器自动插入 retain/release 调用。但临时对象何时释放是不确定的。
    //
    // @autoreleasepool { ... } 创建了一个"自动释放池":
    //   进入块: 创建新的释放池
    //   离开块: 池内所有标记为 autorelease 的对象被 release
    //
    // 类比 C++:
    //   @autoreleasepool { }  ≈  { /* 作用域结束时析构所有智能指针 */ }
    //
    // 为什么需要它？Metal API 会创建大量临时 ObjC 对象。没有 autoreleasepool，
    // 这些对象会堆积到顶层释放池，可能造成内存峰值。显式写 @autoreleasepool
    // 确保每个 Metal 操作完成后立即释放临时对象。
    //
    // 语法注意: @autoreleasepool 不是函数调用，是编译器指令 (类似 #pragma)
    @autoreleasepool {
        // ---- 第1步: 获取默认 GPU 设备 ----
        // MTLCreateSystemDefaultDevice() 返回系统默认 GPU
        // 在 M2 Pro 的 MacBook Pro 上，这就是 M2 Pro 的集成 GPU
        // 如果有外接显卡 (eGPU)，需要用其他 API 选择设备
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "[Metal] No Metal-capable GPU found\n");
            return nullptr;  // C++ 空指针
        }
        // [[device name] UTF8String] 的执行顺序:
        //   1. [device name]   — 发送 name 消息，返回 NSString* 如 @"Apple M2 Pro"
        //   2. [result UTF8String] — 发送 UTF8String 消息，返回 const char*
        NSLog(@"[Metal] Device: %s", [[device name] UTF8String]);

        // ---- 第2步: 编译 Metal shader ----
        // 两种方式:
        //   A) 预编译: xcrun -sdk macosx metal -c xxx.metal -o xxx.air
        //              xcrun -sdk macosx metallib xxx.air -o xxx.metallib
        //              → 运行时加载 .metallib (快)
        //   B) 运行时编译: newLibraryWithSource: (本项目使用，便于开发和调试)
        //
        // 运行时编译在开发阶段更方便 (修改 .metal 后重新运行即可)
        // 生产环境应该预编译 (启动更快，错误检查更早)
        NSString* source = load_metal_source(metal_source_path);
        if (!source) {
            return nullptr;
        }

        NSError* err = nil;
        // MTLCompileOptions: 设置 shader 编译选项
        // MTLLanguageVersion3_1 对应 Metal 3.1 (macOS 14+)
        MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
        opts.languageVersion = MTLLanguageVersion3_1;

        id<MTLLibrary> library = [device newLibraryWithSource:source
                                                      options:opts
                                                        error:&err];
        if (!library) {
            fprintf(stderr, "[Metal] Shader compilation failed: %s\n",
                    [[err localizedDescription] UTF8String]);
            return nullptr;
        }

        // ---- 第3步: 创建 pipeline state ----
        // 从 library 中取出每个 kernel 函数，编译为可执行的 pipeline state
        id<MTLComputePipelineState> naive_pso = create_pipeline(device, library, @"naive_matmul");
        id<MTLComputePipelineState> tiled_pso = create_pipeline(device, library, @"tiled_matmul");
        id<MTLComputePipelineState> simd_pso  = create_pipeline(device, library, @"simd_matmul");

        if (!naive_pso && !tiled_pso && !simd_pso) {
            fprintf(stderr, "[Metal] No usable kernels found\n");
            return nullptr;
        }

        // ---- 第4步: 创建命令队列 ----
        // CommandQueue 是 CPU 向 GPU 提交渲染/计算命令的通道
        // 通常整个程序只创建一个，复用
        id<MTLCommandQueue> queue = [device newCommandQueue];

        // ---- 第5步: 组装 context ----
        // 用 C++ new 在堆上分配 context，返回裸指针给调用方
        MetalContext* ctx = new MetalContext();
        ctx->device          = device;
        ctx->queue           = queue;
        ctx->library         = library;
        ctx->naive_pipeline  = naive_pso;
        ctx->tiled_pipeline  = tiled_pso;
        ctx->simd_pipeline   = simd_pso;

        // 查询设备能力 (这些信息对后续优化有用)
        ctx->max_threads_per_threadgroup = (int)device.maxThreadsPerThreadgroup.width;
        ctx->threadgroup_memory_size     = (int)device.maxThreadgroupMemoryLength;

        NSLog(@"[Metal] Init complete. Max threads/threadgroup: %d, "
              "Threadgroup memory: %d bytes",
              ctx->max_threads_per_threadgroup,
              ctx->threadgroup_memory_size);

        return ctx;
    } // @autoreleasepool 结束: 临时 ObjC 对象在这里被释放
}

// =============================================================================
// metal_release: 释放 Metal 上下文
// 简单的 C++ delete，ObjC 对象由 ARC 自动管理
// =============================================================================
void metal_release(MetalContext* ctx) {
    if (!ctx) return;
    // delete 会释放 C++ struct 的内存
    // struct 里的 id<...> 成员由 ARC 自动 release (不需要手动写)
    delete ctx;
    NSLog(@"[Metal] Context released");
}

// =============================================================================
// metal_gpu_name: 获取 GPU 名称 (例如 "Apple M2 Pro")
//
// 注意这个函数的实现: 它把 ObjC NSString 转换为 C 的 const char*
// 调用方 (纯 C++ 代码) 可以安全使用返回的 C 字符串
// =============================================================================
const char* metal_gpu_name(MetalContext* ctx) {
    if (!ctx) return "N/A";
    static char name_buf[256];
    // ★ 桥接: ObjC NSString → C 字符串
    strncpy(name_buf, [[ctx->device name] UTF8String], sizeof(name_buf) - 1);
    name_buf[sizeof(name_buf) - 1] = '\0';
    return name_buf;
}

void metal_threadgroup_info(MetalContext* ctx,
                             int* max_threads_per_threadgroup,
                             int* threadgroup_memory_size) {
    if (!ctx) {
        *max_threads_per_threadgroup = 0;
        *threadgroup_memory_size = 0;
        return;
    }
    *max_threads_per_threadgroup = ctx->max_threads_per_threadgroup;
    *threadgroup_memory_size = ctx->threadgroup_memory_size;
}

// =============================================================================
// metal_matmul_dispatch: GPU 矩阵乘法的核心 — 设置 + 分发 + 等待 + 读回
//
// 这是每个 Metal 计算任务的标准流程，任何 Metal 计算程序都遵循这个模式:
//
//   CPU 侧 (这个函数):                    GPU 侧 (shader):
//   ────────────────                      ────────────
//   1. 创建 MTLBuffer (GPU 内存)          ↓
//   2. 拷贝 A, B 数据到 buffer            等待命令
//   3. 创建 MTLCommandBuffer             ↓
//   4. 创建 MTLComputeCommandEncoder     收到 kernel 和参数
//   5. 设置 pipeline state               ↓
//   6. 绑定 buffer 参数                  几千个线程并行执行 kernel
//   7. 设置 threadgroup memory (可选)     ↓
//   8. dispatchThreads (分发线程!)        写入结果到 buffer C
//   9. endEncoding                       ↓
//   10. commit (提交给 GPU!)             完成
//   11. waitUntilCompleted (等待完成)
//   12. memcpy 读回结果
//
//   关键点: 步骤 10 (commit) 之前，GPU 不会开始工作！
//   所有这些设置都是"录制"，commit 后才提交给 GPU 执行。
// =============================================================================
static int metal_matmul_dispatch(MetalContext* ctx,
                                  id<MTLComputePipelineState> pso,
                                  const float* A, const float* B, float* C,
                                  int M, int K, int N,
                                  int threads_per_tg,
                                  bool use_threadgroup_mem)
{
    if (!ctx || !pso) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -1;

    @autoreleasepool {
        size_t size_A = (size_t)M * K * sizeof(float);
        size_t size_B = (size_t)K * N * sizeof(float);
        size_t size_C = (size_t)M * N * sizeof(float);

        // ---- 第1步: 创建 GPU 缓冲区 ----
        //
        // MTLResourceStorageModeShared 的含义:
        //   CPU 和 GPU 共享同一块物理内存。在 Apple Silicon (M2 Pro) 上，
        //   这是"统一内存架构"的基础 —— CPU 和 GPU 访问同一个物理 RAM。
        //   所以 [newBufferWithBytes:A ...] 实际上是"映射"而非"拷贝"。
        //
        // 另外两种模式:
        //   MTLResourceStorageModePrivate: 只有 GPU 能访问 (性能最高)
        //   MTLResourceStorageModeManaged:  CPU 和 GPU 各自有拷贝 (Intel Mac 需要)
        //
        MTLResourceOptions opts = MTLResourceStorageModeShared;

        // newBufferWithBytes:length:options:
        //   从 CPU 内存 A 初始化 buffer 内容，一步完成"分配+拷贝"
        id<MTLBuffer> bufA = [ctx->device newBufferWithBytes:A length:size_A options:opts];
        id<MTLBuffer> bufB = [ctx->device newBufferWithBytes:B length:size_B options:opts];

        // newBufferWithLength:options:
        //   只分配 GPU 内存，不初始化内容 (结果由 GPU 写入)
        id<MTLBuffer> bufC = [ctx->device newBufferWithLength:size_C options:opts];

        if (!bufA || !bufB || !bufC) {
            fprintf(stderr, "[Metal] Buffer allocation failed\n");
            return -1;
        }

        // ---- 第2步: 创建命令缓冲区和计算编码器 ----
        //
        // MTLCommandBuffer: 一组 GPU 命令的容器。可以有多个 encoder。
        //   类比: 一个 "工作清单"（要做的事情列表）
        //
        // MTLComputeCommandEncoder: 专门编码"计算"类命令 (还有渲染 encoder)
        //   类比: 填写工作清单具体条目的笔
        //
        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        // ---- 第3步: 设置 kernel 和参数 ----
        //
        // 参数绑定跟随 shader 中的 buffer(N) 声明:
        //   kernel void naive_matmul(
        //       device const float* A  [[buffer(0)]],   ← 对应 atIndex:0
        //       device const float* B  [[buffer(1)]],   ← 对应 atIndex:1
        //       device float*       C  [[buffer(2)]],   ← 对应 atIndex:2
        //       constant uint& M       [[buffer(3)]],   ← 对应 atIndex:3
        //       constant uint& K       [[buffer(4)]],   ← 对应 atIndex:4
        //       constant uint& N       [[buffer(5)]],   ← 对应 atIndex:5
        //   )
        // CPU 侧通过 atIndex 与 GPU 侧的 buffer(N) 一一对应！

        // setComputePipelineState: 设置要执行的 kernel
        [encoder setComputePipelineState:pso];

        // setBuffer:offset:atIndex: 绑定 MTLBuffer 到 shader 参数
        [encoder setBuffer:bufA offset:0 atIndex:0];  // → shader buffer(0): A
        [encoder setBuffer:bufB offset:0 atIndex:1];  // → shader buffer(1): B
        [encoder setBuffer:bufC offset:0 atIndex:2];  // → shader buffer(2): C

        // setBytes:length:atIndex: 直接传值 (适合小数据)
        // 编译器会在幕后创建一个小的临时 buffer
        // shader 中 constant uint& 对应这里的 setBytes
        uint m = (uint)M, k = (uint)K, n = (uint)N;
        [encoder setBytes:&m length:sizeof(uint) atIndex:3];  // → shader buffer(3): M
        [encoder setBytes:&k length:sizeof(uint) atIndex:4];  // → shader buffer(4): K
        [encoder setBytes:&n length:sizeof(uint) atIndex:5];  // → shader buffer(5): N

        // ---- 第4步: 设置 threadgroup memory (仅分块版本) ----
        //
        // threadgroup memory 是 on-chip SRAM，类似 CPU 的 L1 cache。
        // 但它是"软件管理的" —— 开发者决定存什么、什么时候同步。
        //
        // 每个 threadgroup buffer 在 shader 中有单独的索引:
        //   threadgroup float* As [[threadgroup(0)]];
        //   threadgroup float* Bs [[threadgroup(1)]];
        //
        // CPU 侧必须为每个索引设置对应的内存大小！
        // 如果只设置 index 0 不设置 index 1 → Bs 的内存为 0，导致越界！
        // 这是之前 GPU tiled 版本计算错误的原因。
        if (use_threadgroup_mem) {
            NSUInteger tile_mem = 16 * 16 * sizeof(float);  // 16x16 floats = 1KB
            [encoder setThreadgroupMemoryLength:tile_mem atIndex:0];  // As
            [encoder setThreadgroupMemoryLength:tile_mem atIndex:1];  // Bs
        }

        // ---- 第5步: 计算 grid 和 threadgroup 尺寸 ----
        //
        // Metal 使用 3D 尺寸 (支持 3D 计算任务):
        //   MTLSizeMake(width, height, depth)
        //
        // grid: 覆盖整个输出的"线程网"
        //   我们要计算 M x N 个输出元素，所以 grid = (N, M, 1)
        //   注意: grid.x = N (列), grid.y = M (行)，因为 GPU 线程是 (x,y) 布局
        //
        // threadgroup: 每个协作组的线程数
        //   threads_per_tg x threads_per_tg = 16x16 = 256 线程
        //   这是 Apple GPU 的合理值 (最大 1024)
        //
        MTLSize gridSize = MTLSizeMake(N, M, 1);
        MTLSize tgSize   = MTLSizeMake(threads_per_tg, threads_per_tg, 1);

        // ---- 第6步: 分发线程 ★ 关键! ----
        //
        // dispatchThreads:threadsPerThreadgroup:
        //   告诉 GPU: "用 tgSize 大小的 threadgroup 去覆盖 gridSize 这么大的任务"
        //
        // 对于 M=1024, N=1024, tgSize=16x16:
        //   GPU 会创建 (1024/16) x (1024/16) = 64x64 = 4096 个 threadgroup
        //   每个 threadgroup 有 256 个线程
        //   总共 4096 x 256 = 1,048,576 个线程并发执行
        //   每个线程执行 kernel 函数的一次实例
        //
        // GPU 的硬件调度器自动管理这些线程的分发和切换
        [encoder dispatchThreads:gridSize threadsPerThreadgroup:tgSize];

        // ---- 第7步: 结束编码，提交，等待 ----
        //
        // endEncoding: 完成"填写工作清单"
        [encoder endEncoding];

        // commit: 把工作清单提交给 GPU 执行队列
        // ★ 在此之前 GPU 不会开始任何工作！
        [cmdBuf commit];

        // waitUntilCompleted: CPU 阻塞等待 GPU 完成
        // 这是一个同步点 —— CPU 不会继续往下走直到 GPU 完成所有工作
        //
        // 同步 vs 异步:
        //   同步 (这里): commit + waitUntilCompleted → 简单，适合测试和简单任务
        //   异步 (生产): commit + addCompletedHandler → 不阻塞 CPU，适合流水线
        [cmdBuf waitUntilCompleted];

        // 检查 GPU 执行是否有错误
        if (cmdBuf.error) {
            fprintf(stderr, "[Metal] Command buffer error: %s\n",
                    [[cmdBuf.error localizedDescription] UTF8String]);
            return -1;
        }

        // ---- 第8步: 读回结果 ----
        //
        // [bufC contents] 返回指向 GPU buffer 的 CPU 可访问指针
        // 在 Apple Silicon 统一内存架构上，这是零拷贝的 ——
        // 直接返回 GPU 写入的物理内存地址
        memcpy(C, [bufC contents], size_C);

        return 0;
    } // @autoreleasepool: 临时 ObjC 对象在这里释放
}

// =============================================================================
// 公开的矩阵乘法接口
//
// 这两个就是给 test_metal.cpp 调用的函数。
// 调用方只需要 #include "metal_matmul.h"，不需要知道 ObjC。
// =============================================================================

int metal_matmul_naive(MetalContext* ctx,
                        const float* A, const float* B, float* C,
                        int M, int K, int N) {
    return metal_matmul_dispatch(ctx, ctx->naive_pipeline,
                                  A, B, C, M, K, N,
                                  /*threads_per_tg=*/16,
                                  /*use_threadgroup_mem=*/false);
}

int metal_matmul_tiled(MetalContext* ctx,
                        const float* A, const float* B, float* C,
                        int M, int K, int N) {
    return metal_matmul_dispatch(ctx, ctx->tiled_pipeline,
                                  A, B, C, M, K, N,
                                  /*threads_per_tg=*/16,
                                  /*use_threadgroup_mem=*/true);
}

// =============================================================================
// simdgroup 矩阵乘法 — 独立的 dispatch 路径
//
// 与 naive/tiled 不同，simd 版本使用 dispatchThreadgroups (而非 dispatchThreads)。
// 原因: simd_matmul kernel 使用 threadgroup_position_in_grid 来确定 8×8 块位置，
//       所以需要精确控制 threadgroup 的数量和布局。
// =============================================================================

int metal_matmul_simd(MetalContext* ctx,
                       const float* A, const float* B, float* C,
                       int M, int K, int N) {
    if (!ctx || !ctx->simd_pipeline) return -1;
    if (M <= 0 || K <= 0 || N <= 0) return -1;

    @autoreleasepool {
        size_t size_A = (size_t)M * K * sizeof(float);
        size_t size_B = (size_t)K * N * sizeof(float);
        size_t size_C = (size_t)M * N * sizeof(float);

        MTLResourceOptions opts = MTLResourceStorageModeShared;

        id<MTLBuffer> bufA = [ctx->device newBufferWithBytes:A length:size_A options:opts];
        id<MTLBuffer> bufB = [ctx->device newBufferWithBytes:B length:size_B options:opts];
        id<MTLBuffer> bufC = [ctx->device newBufferWithLength:size_C options:opts];

        if (!bufA || !bufB || !bufC) {
            fprintf(stderr, "[Metal] Buffer allocation failed\n");
            return -1;
        }

        id<MTLCommandBuffer> cmdBuf = [ctx->queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];

        [encoder setComputePipelineState:ctx->simd_pipeline];
        [encoder setBuffer:bufA offset:0 atIndex:0];
        [encoder setBuffer:bufB offset:0 atIndex:1];
        [encoder setBuffer:bufC offset:0 atIndex:2];

        uint m = (uint)M, k = (uint)K, n = (uint)N;
        [encoder setBytes:&m length:sizeof(uint) atIndex:3];
        [encoder setBytes:&k length:sizeof(uint) atIndex:4];
        [encoder setBytes:&n length:sizeof(uint) atIndex:5];

        // threadgroup memory: 两个 8×8 tile = 2 × 64 × 4 = 512 bytes
        NSUInteger tile_mem = 8 * 8 * sizeof(float);
        [encoder setThreadgroupMemoryLength:tile_mem atIndex:0];  // As
        [encoder setThreadgroupMemoryLength:tile_mem atIndex:1];  // Bs

        // ---- 使用 dispatchThreadgroups (不是 dispatchThreads!) ----
        //
        // 每个 threadgroup 是 32 个线程 (1 simdgroup)，处理 8×8 输出块。
        // threadgroup 数量 = 向上取整的 8×8 块数
        // threadsPerThreadgroup = (32, 1, 1)
        //
        MTLSize tgCount  = MTLSizeMake((N + 7) / 8, (M + 7) / 8, 1);
        MTLSize tgSize   = MTLSizeMake(32, 1, 1);

        [encoder dispatchThreadgroups:tgCount threadsPerThreadgroup:tgSize];

        [encoder endEncoding];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        if (cmdBuf.error) {
            fprintf(stderr, "[Metal] Command buffer error: %s\n",
                    [[cmdBuf.error localizedDescription] UTF8String]);
            return -1;
        }

        memcpy(C, [bufC contents], size_C);
        return 0;
    }
}
