# Metal GPU 编程学习指南

本文档配合项目代码讲解 Metal 框架和 GPU 编程的核心概念，专为 C++ 开发者编写。

---

## 1. 为什么 Metal 需要 Objective-C？

### Apple 的设计选择

Metal 是 Apple 在 2014 年推出的低开销 GPU 框架。Apple 从第一天起就只提供了两个官方语言绑定：

| 语言 | 文件后缀 | Metal API 访问方式 |
|------|---------|-------------------|
| **Objective-C** | `.m` | `#import <Metal/Metal.h>` |
| **Swift** | `.swift` | `import Metal` |
| C++ | `.cpp` | **没有官方 API！** |

后来 Apple 发布了 [metal-cpp](https://developer.apple.com/metal/cpp/)（一个 C++ 封装库），但它本质上还是调用 ObjC 方法。

### 所以我们必须用 `.mm` 文件

`.mm` = **Objective-C++**，这是 clang 编译器的特殊模式：

```
┌──────────────────────────────────────────────────────┐
│              .mm 文件 = 混合语言文件                    │
│                                                      │
│  #include <cstdio>        ← C++ 标准库                │
│  #include <vector>        ← C++ STL                  │
│  #import <Metal/Metal.h>  ← ObjC 框架                │
│                                                      │
│  // C++ 代码:                                        │
│  struct Foo { int x; };                              │
│  Foo* p = new Foo();                                 │
│  delete p;                                           │
│                                                      │
│  // ObjC 代码:                                       │
│  id<MTLDevice> device = MTLCreateSystemDefaultDevice();│
│  NSString* name = @"hello";                          │
│  [device newCommandQueue];                           │
│                                                      │
│  // 两者可以混合在同一函数中！                          │
│  fprintf(stderr, "GPU: %s\n", [[device name] UTF8String]);│
│       ↑ C++                    ↑ ObjC                   │
└──────────────────────────────────────────────────────┘
```

clang 看到 `.mm` 后缀时，会同时启用 C++ 和 ObjC 的解析器，两种语法在同一文件中都合法。

---

## 2. ObjC 速览（写给 C++ 开发者）

### 2.1 类和方法

```
C++:                              ObjC:
──────────────────────────────────────────────────
class Dog {                       @interface Dog : NSObject
public:                           - (void)bark;
  void bark();                    - (void)eat:(NSString*)food;
  void eat(string food);          @end
};
                                  // 实现
Dog* d = new Dog();               @implementation Dog
d->bark();                        - (void)bark { NSLog(@"Woof"); }
d->eat("bone");                   - (void)eat:(NSString*)food {
delete d;                             NSLog(@"Eating %@", food);
                                  }
                                  @end
                                  Dog* d = [[Dog alloc] init];
                                  [d bark];
                                  [d eat:@"bone"];
                                  // ARC 自动释放 d
```

### 2.2 消息传递语法（`[ ]` 括号）

这是 ObjC 最显著的特征：

```
C++:  obj->method(arg1, arg2)
ObjC: [obj method:arg1 withParam:arg2]
```

方法名被"打散"穿插在参数之间，这被称为"交织参数"（interleaved arguments）。

```
// 无参数:
C++:  device->newCommandQueue()
ObjC: [device newCommandQueue]

// 一个参数:
C++:  encoder->setBuffer(buf, 0, 0)
ObjC: [encoder setBuffer:buf offset:0 atIndex:0]
//                 ↑方法名第1段  ↑参1  ↑第2段  ↑参2  ↑第3段  ↑参3

// 完整方法名是: setBuffer:offset:atIndex:
// 这是 ObjC 特有的风格 —— 方法名本身带有参数说明
```

### 2.3 `id` 类型和协议

```
C++:                              ObjC:
──────────────────────────────────────────────────
class IDevice {                    @protocol MTLDevice
  virtual Buffer* newBuffer() = 0; - (id<MTLBuffer>)newBufferWithLength:(NSUInteger)len;
};                                 @end

IDevice* dev;    // 接口指针        id<MTLDevice> dev;  // 遵循 MTLDevice 协议的任意对象
```

`id` ≈ `void*`（万能对象指针），但可以附加协议约束：
- `id<MTLDevice>`: 指向遵循 `MTLDevice` 协议的对象
- `id<MTLDevice, MTLCommandQueue>`: 指向遵循两个协议的对象

**关键区别**: `id` 是动态类型的，编译时不检查方法是否存在。给 `id` 发送任何消息编译器都不会报错（但运行时会检查）。

### 2.4 内存管理 (ARC)

```
C++:                              ObjC (ARC):
──────────────────────────────────────────────────
// 手动管理                        // 自动引用计数
Foo* p = new Foo();               Foo* p = [[Foo alloc] init];
// ... 使用 p ...                   // ... 使用 p ...
delete p;                         // 离开作用域，ARC 自动 release
```

ARC (Automatic Reference Counting) 编译时自动插入 `retain`/`release` 调用。不需要手动 `delete`。

### 2.5 `@autoreleasepool`

```
相当于 C++ 的:
{
    std::vector<std::unique_ptr<TempObj>> pool;
    // ... 创建很多临时对象 ...
} // 作用域结束，所有临时对象析构

ObjC 写法:
@autoreleasepool {
    // ... 创建很多临时 ObjC 对象 ...
} // 离开时，池内所有临时对象被 release
```

Metal API 调用会产生大量临时对象，所以每次 GPU 操作都应该包裹在 `@autoreleasepool` 中。

### 2.6 C ↔ ObjC 字符串转换

```
// C → ObjC
const char* c_str = "hello";
NSString* ns_str = [NSString stringWithUTF8String:c_str];

// ObjC → C
NSString* ns_str = @"world";
const char* c_str = [ns_str UTF8String];
```

本项目中的模式（大量出现）:

```cpp
// metal_matmul.mm 中:
fprintf(stderr, "GPU: %s\n", [[device name] UTF8String]);
//                            └──────────────┬──────────────┘
//                            ObjC 调用，返回 NSString*
//                           └─────────────────┬───────────────┘
//                           再调用 UTF8String 转为 C 字符串
```

---

## 3. Metal 编程模型

### 3.1 核心对象

```
┌──────────────────────────────────────────────┐
│                 Metal 对象层次                  │
│                                              │
│  MTLDevice        ← GPU 硬件抽象               │
│    │                                         │
│    ├─ MTLCommandQueue  ← 命令提交通道           │
│    │   │                                     │
│    │   └─ MTLCommandBuffer ← 一组 GPU 命令      │
│    │       │                                 │
│    │       ├─ MTLComputeCommandEncoder ← 计算   │
│    │       │   · setPipelineState()            │
│    │       │   · setBuffer()                   │
│    │       │   · dispatchThreads()             │
│    │       │                                   │
│    │       └─ MTLRenderCommandEncoder ← 渲染    │
│    │                                         │
│    ├─ MTLLibrary      ← 编译后的 shader 库     │
│    │   └─ MTLFunction  ← 单个 kernel 函数       │
│    │       └─ MTLComputePipelineState ← 可执行   │
│    │                                         │
│    └─ MTLBuffer       ← GPU 内存缓冲区          │
└──────────────────────────────────────────────┘
```

### 3.2 GPU 计算的标准流程

每个 Metal 计算任务都遵循这个模式：

```
1. 创建 MTLBuffer (GPU 内存)
2. 从 CPU 拷贝输入数据到 buffer
3. 创建 MTLCommandBuffer (命令清单)
4. 创建 MTLComputeCommandEncoder (开始"录制"命令)
5. 设置 pipeline state (选择哪个 kernel)
6. 绑定参数 (buffer(0), buffer(1), ...)
7. 设置 threadgroup memory (如果需要)
8. dispatchThreads (指定线程数量)
9. endEncoding (结束录制)
10. commit (提交到 GPU!)
11. waitUntilCompleted (等待完成)
12. 从 buffer 读回结果
```

### 3.3 同步 vs 异步

```
// 同步模式 (本项目的做法):
// 简单，适合测试和简单任务
[cmdBuf commit];
[cmdBuf waitUntilCompleted];  // CPU 阻塞等待
memcpy(result, [bufC contents], size);  // 安全读回

// 异步模式 (生产环境):
// 不阻塞 CPU，适合复杂流水线
[cmdBuf addCompletedHandler:^(id<MTLCommandBuffer> buf) {
    memcpy(result, [bufC contents], size);
    // 信号量或其他通知机制
}];
[cmdBuf commit];
// CPU 继续做其他事...
```

---

## 4. GPU 内存层级

从快到慢（也从小到大）：

| 内存类型 | 物理位置 | 大小 | 延迟 | 可见范围 |
|---------|---------|------|------|---------|
| **Register** | GPU 核心内 | 每线程几十个 | 0 cycle | 单线程 |
| **Threadgroup** | GPU 芯片 SRAM | 32KB/threadgroup | ~20 cycles | threadgroup 内 |
| **Device (VRAM)** | 独立显存或统一内存 | 几 GB | ~200-400 cycles | 所有线程 |
| **System RAM** | 主机内存 | 几十 GB | ~500+ cycles | CPU |

**类比 CPU**:
- Threadgroup memory ≈ L1 cache（但由软件管理！）
- Device memory ≈ 主内存 RAM

**在 Apple Silicon (M1/M2/M3)** 上：
- CPU 和 GPU 共享同一块物理内存（统一内存架构）
- 所以 `MTLResourceStorageModeShared` 实际上是零拷贝的
- "拷贝数据到 GPU" ≈ 传递一个指针

---

## 5. GPU 线程模型

### 5.1 三级层级

```
grid (全部线程)
├── threadgroup (0)           ← 可以共享 threadgroup memory
│   ├── thread 0              ← 每个线程执行相同的 kernel
│   ├── thread 1
│   ├── ...
│   └── thread 255            ← 16x16 threadgroup = 256 threads
├── threadgroup (1)
│   └── ...
├── threadgroup (2)
│   └── ...
└── ...
```

### 5.2 关键概念：线程标识

```metal
kernel void my_kernel(
    uint2 gid [[thread_position_in_grid]],           // 全局位置
    uint2 tid [[thread_position_in_threadgroup]],    // 组内位置
    uint2 tgid [[threadgroup_position_in_grid]])     // 组位置
{
    // gid = 线程在整个 grid 中的位置 (类似全局 ID)
    // tid = 线程在当前 threadgroup 中的位置 (0..15)
    // tgid = 当前 threadgroup 在 grid 中的位置

    // 关系: gid = tgid * threadgroup_size + tid
}
```

对于我们的矩阵乘法：
- `gid.y` = 输出矩阵的行
- `gid.x` = 输出矩阵的列
- 线程负责计算 `C[gid.y][gid.x]`

### 5.3 Threadgroup Barrier

```metal
threadgroup_barrier(mem_flags::mem_threadgroup);
```

这确保了：
1. **同步**: threadgroup 内所有线程都到达这个 barrier 才继续
2. **内存可见**: barrier 前的所有 threadgroup memory 写入对 barrier 后的读可见

**经典双 barrier 模式**（tiled_matmul 中）：

```metal
// 加载阶段: 所有线程协作加载 tile 数据
As[tid] = A[...];   // 线程各自加载
Bs[tid] = B[...];

threadgroup_barrier(...);  // ★ Barrier 1: 确保加载完成

// 计算阶段: 从 threadgroup memory 读取
sum += As[...] * Bs[...];

threadgroup_barrier(...);  // ★ Barrier 2: 确保计算完成才能覆盖 tile
// 下一轮迭代开始, 再次加载到 As/Bs...
```

**为什么需要 Barrier 2？**
没有它，快的线程在下一轮迭代会覆盖 As/Bs，而慢的线程还在读旧的 As/Bs 数据。

---

## 6. 本项目的实现层次

```
test_metal.cpp        ← 纯 C++ 测试程序
    │                    #include "metal_matmul.h"
    ├── metal_matmul.h   ← 纯 C++ 接口 (不透明指针)
    │
    └── metal_matmul.mm  ← ObjC++ 实现
            │               #import <Metal/Metal.h>
            │               封装 Metal API 调用
            │
            └── metal_shaders.metal  ← GPU shader
                    │                   #include <metal_stdlib>
                    ├── naive_matmul
                    └── tiled_matmul
```

**分离原则**:
- `.h` 只暴露 C 类型（`float*`，`int`，`struct MetalContext*`）
- `.mm` 内部使用 ObjC 类型（`id<MTLDevice>`, `NSString*`）
- `.cpp` 调用方完全不知道 ObjC 的存在

---

## 7. 性能分析：为什么 GPU 更快？

以 2048x2048 矩阵乘法为例：

| 实现 | GFlops | 说明 |
|-----|--------|------|
| CPU Naive | 0.6 | 三重循环，cache miss 严重 |
| CPU Best (SIMD+MT) | 146 | NEON SIMD + 8核并行 |
| GPU Naive | 369 | 百万线程并行，但受 VRAM 带宽限制 |
| GPU Tiled | 669 | Threadgroup memory 减少 VRAM 访问 |

**GPU 加速的本质**:
1. **大量并行**：M2 Pro 有 19 个 GPU 核心，每个核心可运行 ~1024 个线程 = 近 2 万个线程并发
2. **内存带宽**：M2 Pro 统一内存带宽 ~200 GB/s（CPU 访问主存约 100 GB/s）
3. **Threadgroup memory**：on-chip SRAM 大幅减少全局内存访问

---

## 8. 下一步学习

1. **simdgroup 操作**: Metal 3.0+ 的 `simdgroup_multiply_accumulate` 直接使用 GPU 矩阵硬件
2. **异步执行**: 使用 `addCompletedHandler` 实现非阻塞 GPU 调用
3. **双缓冲**: 利用 GPU 执行和 CPU 数据准备的时间重叠
4. **Metal 调试工具**: Xcode 的 GPU Frame Capture、Shader Debugger
5. **移植到 CUDA**: 相同的概念（grid/block/thread、shared memory）映射到 CUDA 术语

---

## 参考资料

- [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf)
- [Metal Compute Programming Guide](https://developer.apple.com/documentation/metal)
- [Metal Best Practices Guide](https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/)
- [Apple metal-cpp](https://developer.apple.com/metal/cpp/)
