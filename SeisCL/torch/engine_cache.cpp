#include "engine_cache.h"

#include <algorithm>

namespace seiscl_torch {

EngineHandle *EngineCache::get_or_create(const CacheKey &key, bool *was_hit) {
    std::lock_guard<std::mutex> lock(mu_);

    auto it = handles_.find(key);
    if (it != handles_.end()) {
        *was_hit = it->second->built;
        lru_.remove(key);
        lru_.push_front(key);
        return it->second.get();
    }

    *was_hit = false;
    auto handle = std::unique_ptr<EngineHandle>(new EngineHandle());
    handle->key = key;
    EngineHandle *raw = handle.get();
    handles_.emplace(key, std::move(handle));
    lru_.push_front(key);
    trim();
    return raw;
}

void EngineCache::evict(const CacheKey &key) {
    std::lock_guard<std::mutex> lock(mu_);
    handles_.erase(key);
    lru_.remove(key);
}

void EngineCache::set_max_size(std::size_t n) {
    std::lock_guard<std::mutex> lock(mu_);
    max_size_ = n;
    trim();
}

std::size_t EngineCache::size() {
    std::lock_guard<std::mutex> lock(mu_);
    return handles_.size();
}

void EngineCache::clear() {
    std::lock_guard<std::mutex> lock(mu_);
    handles_.clear();
    lru_.clear();
}

// Caller holds mu_. Never evicts the front entry: that is the handle the
// in-flight call is about to use.
void EngineCache::trim() {
    while (handles_.size() > max_size_ && lru_.size() > 1) {
        CacheKey victim = lru_.back();
        lru_.pop_back();
        handles_.erase(victim);
    }
}

EngineCache &global_engine_cache() {
    static EngineCache cache;
    return cache;
}

}  // namespace seiscl_torch
