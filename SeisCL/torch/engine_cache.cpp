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

std::unique_ptr<EngineHandle> EngineCache::rekey(const CacheKey &from,
                                                 const CacheKey &to) {
    std::lock_guard<std::mutex> lock(mu_);
    if (from == to) return nullptr;
    auto it = handles_.find(from);
    if (it == handles_.end()) return nullptr;

    std::unique_ptr<EngineHandle> handle = std::move(it->second);
    handles_.erase(it);
    lru_.remove(from);

    // Any handle already sitting on the destination key is displaced, not
    // destroyed here -- see the header comment on why the caller must check
    // pending_valid on what this returns.
    std::unique_ptr<EngineHandle> displaced;
    auto dst = handles_.find(to);
    if (dst != handles_.end()) {
        displaced = std::move(dst->second);
        handles_.erase(dst);
        lru_.remove(to);
    }

    handle->key = to;
    handles_.emplace(to, std::move(handle));
    lru_.push_front(to);
    return displaced;
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
// in-flight call is about to use. Also never evicts a handle holding an
// unwritten checkpoint, whose buffers are the only copy of a forward pass
// some later backward call still needs -- such a handle stays until that
// backward consumes it, so the cache can temporarily exceed max_size_.
void EngineCache::trim() {
    auto it = lru_.end();
    while (handles_.size() > max_size_ && lru_.size() > 1) {
        if (it == lru_.begin()) break;
        --it;
        auto found = handles_.find(*it);
        if (found != handles_.end() && found->second->pending_valid) continue;
        CacheKey victim = *it;
        it = lru_.erase(it);
        handles_.erase(victim);
    }
}

EngineCache &global_engine_cache() {
    static EngineCache cache;
    return cache;
}

}  // namespace seiscl_torch
