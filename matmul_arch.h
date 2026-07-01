// =============================================================================
// matmul_arch.h — 平台检测 & SIMD 抽象层
//
// 只负责一件事：定义当前平台的 SIMD 类型和基础操作。
// 不包含任何矩阵乘法逻辑。
//
// 定义的宏 (编译期常量):
//   MATMUL_ARCH_X86_AVX    — x86 with AVX (256-bit)
//   MATMUL_ARCH_X86_SSE    — x86 with SSE only (128-bit)
//   MATMUL_ARCH_ARM_NEON   — ARM with NEON (128-bit)
//   MATMUL_ARCH_APPLE      — Apple 平台 (macOS/iOS)
//   MATMUL_HAS_METAL       — Apple 平台且可以链接 Metal
//   MATMUL_HAS_ACCELERATE  — 可以链接 Accelerate.framework
//   MATMUL_ARCH_GENERIC    — 纯标量回退
//
// 定义的类型:
//   matmul_f32    — 平台最优 SIMD 向量
//   matmul_f32x2  — 双倍宽度
//
// 定义的函数 (namespace matmul_simd):
//   load, set1, add, mul, fmadd, horizontal_sum
// =============================================================================

#pragma once

// ---- 第1层: 操作系统 & 编译器检测 ----
#if defined(__APPLE__) && defined(__MACH__)
    #define MATMUL_OS_APPLE 1
    // Apple 上可以通过 __has_include 检测 Metal 和 Accelerate
    #if __has_include(<Metal/Metal.h>)
        #define MATMUL_HAS_METAL 1
    #endif
    #if __has_include(<Accelerate/Accelerate.h>)
        #define MATMUL_HAS_ACCELERATE 1
    #endif
#endif

// ---- 第2层: 指令集架构检测 ----
#if defined(__aarch64__) || defined(__ARM_NEON__)
    // ARM 64-bit (Apple Silicon, 树莓派4+, 鲲鹏等)
    #define MATMUL_ARCH_ARM_NEON 1
    #include <arm_neon.h>

#elif defined(__AVX__)
    // x86 with AVX (256-bit, Sandy Bridge+)
    #define MATMUL_ARCH_X86_AVX 1
    #define MATMUL_ARCH_X86_SSE 1
    #include <immintrin.h>

#elif defined(__SSE__) || defined(_M_X64) || defined(_M_IX86)
    // x86 with SSE only (128-bit)
    #define MATMUL_ARCH_X86_SSE 1
    #include <xmmintrin.h>
    #include <emmintrin.h>

#else
    // 纯标量回退
    #define MATMUL_ARCH_GENERIC 1
#endif

// ---- 第3层: SIMD 类型统一 ----
// 所有平台的向量都叫 matmul_f32

#if defined(MATMUL_ARCH_ARM_NEON)
    typedef float32x4_t      matmul_f32;
    typedef float32x4x2_t    matmul_f32x2;
    #define MATMUL_SIMD_WIDTH 4
    #define MATMUL_SIMD_ALIGN 16

#elif defined(MATMUL_ARCH_X86_AVX)
    typedef __m256           matmul_f32;
    typedef struct { __m256 a, b; } matmul_f32x2;
    #define MATMUL_SIMD_WIDTH 8
    #define MATMUL_SIMD_ALIGN 32

#elif defined(MATMUL_ARCH_X86_SSE)
    typedef __m128           matmul_f32;
    typedef struct { __m128 a, b; } matmul_f32x2;
    #define MATMUL_SIMD_WIDTH 4
    #define MATMUL_SIMD_ALIGN 16

#else
    typedef struct { float data[4]; } matmul_f32;
    typedef struct { matmul_f32 a, b; } matmul_f32x2;
    #define MATMUL_SIMD_WIDTH 4
    #define MATMUL_SIMD_ALIGN 16
#endif

// ---- 第4层: SIMD 操作统一 ----
// 用一个 namespace 抹平 neon/vaddq_f32 vs avx/_mm256_add_ps

namespace matmul_simd {

// 从内存加载
static inline matmul_f32 load(const float* ptr) {
#if defined(MATMUL_ARCH_ARM_NEON)
    return vld1q_f32(ptr);
#elif defined(MATMUL_ARCH_X86_AVX)
    return _mm256_loadu_ps(ptr);
#elif defined(MATMUL_ARCH_X86_SSE)
    return _mm_loadu_ps(ptr);
#else
    matmul_f32 r;
    for (int i = 0; i < 4; i++) r.data[i] = ptr[i];
    return r;
#endif
}

// 广播标量到所有 lane
static inline matmul_f32 set1(float v) {
#if defined(MATMUL_ARCH_ARM_NEON)
    return vdupq_n_f32(v);
#elif defined(MATMUL_ARCH_X86_AVX)
    return _mm256_set1_ps(v);
#elif defined(MATMUL_ARCH_X86_SSE)
    return _mm_set1_ps(v);
#else
    matmul_f32 r;
    for (int i = 0; i < 4; i++) r.data[i] = v;
    return r;
#endif
}

static inline matmul_f32 add(matmul_f32 a, matmul_f32 b) {
#if defined(MATMUL_ARCH_ARM_NEON)
    return vaddq_f32(a, b);
#elif defined(MATMUL_ARCH_X86_AVX)
    return _mm256_add_ps(a, b);
#elif defined(MATMUL_ARCH_X86_SSE)
    return _mm_add_ps(a, b);
#else
    matmul_f32 r;
    for (int i = 0; i < 4; i++) r.data[i] = a.data[i] + b.data[i];
    return r;
#endif
}

static inline matmul_f32 mul(matmul_f32 a, matmul_f32 b) {
#if defined(MATMUL_ARCH_ARM_NEON)
    return vmulq_f32(a, b);
#elif defined(MATMUL_ARCH_X86_AVX)
    return _mm256_mul_ps(a, b);
#elif defined(MATMUL_ARCH_X86_SSE)
    return _mm_mul_ps(a, b);
#else
    matmul_f32 r;
    for (int i = 0; i < 4; i++) r.data[i] = a.data[i] * b.data[i];
    return r;
#endif
}

// 乘加: a * b + c
static inline matmul_f32 fmadd(matmul_f32 a, matmul_f32 b, matmul_f32 c) {
#if defined(MATMUL_ARCH_ARM_NEON)
    return vmlaq_f32(c, a, b);
#elif defined(MATMUL_ARCH_X86_AVX)
    return _mm256_fmadd_ps(a, b, c);
#elif defined(MATMUL_ARCH_X86_SSE)
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#else
    matmul_f32 r;
    for (int i = 0; i < 4; i++) r.data[i] = a.data[i] * b.data[i] + c.data[i];
    return r;
#endif
}

// 水平求和: 把向量内所有 lane 加起来
static inline float horizontal_sum(matmul_f32 v) {
#if defined(MATMUL_ARCH_ARM_NEON)
    return vaddvq_f32(v);
#elif defined(MATMUL_ARCH_X86_AVX)
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);
    sum128 = _mm_hadd_ps(sum128, sum128);
    sum128 = _mm_hadd_ps(sum128, sum128);
    return _mm_cvtss_f32(sum128);
#elif defined(MATMUL_ARCH_X86_SSE)
    __m128 sum = _mm_hadd_ps(v, v);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
#else
    return v.data[0] + v.data[1] + v.data[2] + v.data[3];
#endif
}

} // namespace matmul_simd
