// Process-wide LRU cache of built engines, keyed by CacheKey.
//
// Each entry pins a CUDA context, a set of compiled kernels and a full set
// of device buffers, so the cache is deliberately small: the point is to
// serve a training loop that calls with one or two fixed shapes, not to
// memoize arbitrary geometry.

#ifndef SEISCL_TORCH_ENGINE_CACHE_H
#define SEISCL_TORCH_ENGINE_CACHE_H

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "cache_key.h"
#include "engine_handle.h"

namespace seiscl_torch {

class EngineCache {
  public:
    // Returns the handle for key, creating an empty (unbuilt) one on a miss.
    // *was_hit tells the caller whether to build or refresh.
    EngineHandle *get_or_create(const CacheKey &key, bool *was_hit);

    // Drop a handle, freeing its GPU resources. Used both for LRU eviction
    // and to discard a handle whose build or refresh failed -- a handle
    // whose last operation failed must never serve a later call.
    void evict(const CacheKey &key);

    // Note that the most recently used entry is never trimmed, since the
    // in-flight call holds a raw pointer to it. Setting the size to 0
    // therefore still leaves one entry behind; use clear() to release it.
    void set_max_size(std::size_t n);
    std::size_t size();
    void clear();

  private:
    void trim();

    std::unordered_map<CacheKey, std::unique_ptr<EngineHandle>, CacheKeyHash>
        handles_;
    std::list<CacheKey> lru_;  // most recently used at front
    // 2 rather than 1: a needs_grad=true and a needs_grad=false signature
    // are distinct keys and alternate in finite-difference gradient checks,
    // which a size-1 cache would thrash.
    std::size_t max_size_ = 2;
    std::mutex mu_;
};

EngineCache &global_engine_cache();

}  // namespace seiscl_torch

#endif  // SEISCL_TORCH_ENGINE_CACHE_H
