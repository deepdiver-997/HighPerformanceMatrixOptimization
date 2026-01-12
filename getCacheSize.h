/* cache_size.h */
#ifndef CACHE_SIZE_H_
#define CACHE_SIZE_H_

#include <stddef.h>   /* size_t */

/* 返回 0 表示“获取失败” */
size_t get_L1d_cache_size(void);
size_t get_L2_cache_size(void);



/* ---------- 平台检测 ---------- */
#if defined(_WIN32) || defined(_WIN64)
  #define CACHE_SIZE_WIN32
  #include <windows.h>
#elif defined(__APPLE__) && (defined(__MACH__) || defined(__APPLE__))
  #define CACHE_SIZE_DARWIN
  #include <sys/sysctl.h>
#elif defined(__linux__) || defined(__ANDROID__) || defined(__unix__)
  #define CACHE_SIZE_POSIX
  #include <unistd.h>
  #include <stdio.h>
  #include <stdlib.h>
#else
  #define CACHE_SIZE_UNKNOWN
#endif

/* ---------- 实现 ---------- */
#if defined(CACHE_SIZE_POSIX)
static size_t sysfs_cache_size(int level)
{
    /* 用 getconf 路径，最通用，不依赖 /sys 挂载与否 */
    long sz = 0;
    const char *name = NULL;
    if (level == 1) name = "LEVEL1_DCACHE_SIZE";
    else if (level == 2) name = "LEVEL2_CACHE_SIZE";
    else return 0;

    char cmd[128];
    snprintf(cmd, sizeof(cmd), "getconf -a 2>/dev/null | grep '^%s' | awk '{print $2}'", name);
    FILE *p = popen(cmd, "r");
    if (p) {
        fscanf(p, "%ld", &sz);
        pclose(p);
    }
    return sz > 0 ? (size_t)sz : 0;
}

size_t get_L1d_cache_size(void) { return sysfs_cache_size(1); }
size_t get_L2_cache_size(void)  { return sysfs_cache_size(2); }

#elif defined(CACHE_SIZE_DARWIN)
size_t get_L1d_cache_size(void)
{
    int64_t sz = 0;
    size_t len = sizeof(sz);
    sysctlbyname("hw.l1dcachesize", &sz, &len, NULL, 0);
    return sz > 0 ? (size_t)sz : 0;
}
size_t get_L2_cache_size(void)
{
    int64_t sz = 0;
    size_t len = sizeof(sz);
    sysctlbyname("hw.l2cachesize", &sz, &len, NULL, 0);
    return sz > 0 ? (size_t)sz : 0;
}

#elif defined(CACHE_SIZE_WIN32)
#include <malloc.h>
size_t get_L1d_cache_size(void)
{
    size_t ret = 0;
    DWORD len = 0;
    if (GetLogicalProcessorInformationEx(RelationCache, NULL, &len) == FALSE &&
        GetLastError() == ERROR_INSUFFICIENT_BUFFER)
    {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buf =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)_malloca(len);
        if (buf &&
            GetLogicalProcessorInformationEx(RelationCache, buf, &len))
        {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX p = buf;
            DWORD left = len;
            while (left > 0) {
                if (p->Relationship == RelationCache &&
                    p->Cache.Level == 1 &&
                    p->Cache.Type == CacheData)
                {
                    ret = p->Cache.Size;
                    break;
                }
                left -= p->Size;
                p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)((BYTE*)p + p->Size);
            }
        }
        _freea(buf);
    }
    return ret;
}

size_t get_L2_cache_size(void)
{
    size_t ret = 0;
    DWORD len = 0;
    if (GetLogicalProcessorInformationEx(RelationCache, NULL, &len) == FALSE &&
        GetLastError() == ERROR_INSUFFICIENT_BUFFER)
    {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buf =
            (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)_malloca(len);
        if (buf &&
            GetLogicalProcessorInformationEx(RelationCache, buf, &len))
        {
            PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX p = buf;
            DWORD left = len;
            while (left > 0) {
                if (p->Relationship == RelationCache &&
                    p->Cache.Level == 2)
                {
                    ret = p->Cache.Size;
                    break;
                }
                left -= p->Size;
                p = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)((BYTE*)p + p->Size);
            }
        }
        _freea(buf);
    }
    return ret;
}

#else  /* CACHE_SIZE_UNKNOWN */
size_t get_L1d_cache_size(void) { return 0; }
size_t get_L2_cache_size(void)  { return 0; }
#endif



#endif /* CACHE_SIZE_H_ */
