#pragma once
#include "common.hpp"
#include <dlfcn.h>

namespace bench {
// 原头文件在独立共享库中编译，避免同名 device helper 和缺 include guard
// 引起冲突。RTLD_LOCAL 防止符号进入其他适配器的查找范围；原文件编译失败
// 时 available() 返回 false，由 runner 结合 build manifest 记录原因。
class Legacy {
    void* library_ = nullptr;
    using Launch = int (*)(const void*, const void*, void*, int, int, int, cudaStream_t);
    Launch launch_ = nullptr;
public:
    explicit Legacy(const std::string& path) {
        library_ = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (library_) launch_ = reinterpret_cast<Launch>(dlsym(library_, "legacy_launch"));
    }
    ~Legacy() { if (library_) dlclose(library_); }
    Legacy(const Legacy&) = delete;
    Legacy& operator=(const Legacy&) = delete;
    bool available() const { return launch_ != nullptr; }
    cudaError_t launch(const void* a, const void* b, void* c, int m, int n, int k, cudaStream_t stream) {
        return static_cast<cudaError_t>(launch_(a, b, c, m, n, k, stream));
    }
};
}  // namespace bench
