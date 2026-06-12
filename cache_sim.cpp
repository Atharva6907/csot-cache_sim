// ============================================================================
//  cache_sim.cpp — CSoT'26 Low Latency Track, Week 2
//
//  A correct, flat-SoA, zero-allocation, compile-time-geometry simulator of
//  the two-level cache hierarchy in CACHE_SPEC.md.
//
//  Layout per cache level (per set, WAYS == 8):
//    tag[set][way]   : uint64_t   — flat array, set-major
//    state[set]      : uint8_t    — packed bitfields: bit i = valid[way i],
//                                    bit (8+i) ... actually packed as two
//                                    bytes: valid_mask, dirty_mask (one bit
//                                    per way)
//    lru[set]        : uint32_t   — 8 nibbles, order[0]=MRU way .. order[7]=LRU way
//
//  All state is allocated (zero-initialized, fixed-size arrays / vectors sized
//  once) in on_init(). run() never allocates.
// ============================================================================

#include "cache_sim.hpp"

#include <cstdint>
#include <vector>

namespace {

using csot::CacheStats;
using csot::MemAccess;

// ---------------------------------------------------------------------------
// Compile-time geometry (CACHE_SPEC.md §3)
// ---------------------------------------------------------------------------
constexpr int      WAYS         = 8;

constexpr int      L1_SETS      = 64;
constexpr int      L1_INDEX_BITS = 6;                 // log2(64)
constexpr uint64_t L1_SET_MASK  = L1_SETS - 1;        // 63

constexpr int      L2_SETS      = 512;
constexpr int      L2_INDEX_BITS = 9;                 // log2(512)
constexpr uint64_t L2_SET_MASK  = L2_SETS - 1;        // 511

// LRU order packed into a uint32_t: 8 nibbles, nibble k holds the way index
// at MRU-rank k (rank 0 = most recently used, rank 7 = least recently used).
constexpr uint32_t INITIAL_LRU = 0x76543210u;  // way i starts at rank i

// Extract the way index stored at MRU-rank `rank`.
inline int lru_way_at(uint32_t order, int rank) {
    return static_cast<int>((order >> (rank * 4)) & 0xFu);
}

// Move `way` to MRU (rank 0), shifting everything before it down by one rank.
inline uint32_t lru_touch(uint32_t order, int way) {
    // Fast path: `way` is already MRU (very common — repeated access to the
    // same line). Nothing to do.
    if ((order & 0xFu) == static_cast<uint32_t>(way)) {
        return order;
    }
    // Find the rank currently holding `way`.
    int rank = 1;
    for (; rank < WAYS; ++rank) {
        if (lru_way_at(order, rank) == way) break;
    }
    // Shift ranks [0, rank) right by one nibble (i.e. down to ranks [1, rank]),
    // then place `way` at rank 0.
    uint32_t result = order;
    for (int r = rank; r > 0; --r) {
        uint32_t prev_way = lru_way_at(order, r - 1);
        result &= ~(0xFu << (r * 4));
        result |= prev_way << (r * 4);
    }
    result &= ~0xFu;
    result |= static_cast<uint32_t>(way);
    return result;
}

// The LRU victim is whatever sits at rank WAYS-1.
inline int lru_victim(uint32_t order) {
    return lru_way_at(order, WAYS - 1);
}

// ---------------------------------------------------------------------------
// One cache level: flat struct-of-arrays, indexed [set * WAYS + way].
// ---------------------------------------------------------------------------
template <int SETS>
struct CacheLevel {
    std::vector<uint64_t> tag;          // [SETS * WAYS]
    std::vector<uint8_t>  valid_mask;   // [SETS] — bit w = way w valid
    std::vector<uint8_t>  dirty_mask;   // [SETS] — bit w = way w dirty
    std::vector<uint32_t> lru;          // [SETS] — packed MRU..LRU order

    void init() {
        tag.assign(static_cast<size_t>(SETS) * WAYS, 0);
        valid_mask.assign(SETS, 0);
        dirty_mask.assign(SETS, 0);
        lru.assign(SETS, INITIAL_LRU);
    }

    // Find a way in `set` holding `t`. Returns way index in [0,WAYS) or -1.
    inline int find(int set, uint64_t t) const {
        const uint8_t vmask = valid_mask[set];
        const uint64_t* base = &tag[static_cast<size_t>(set) * WAYS];

        unsigned match = 0;
        for (int w = 0; w < WAYS; ++w) {
            match |= static_cast<unsigned>(base[w] == t) << w;
        }
        match &= vmask;

        return match ? __builtin_ctz(match) : -1;
    }

    inline void touch(int set, int way) {
        lru[set] = lru_touch(lru[set], way);
    }

    inline bool is_valid(int set, int way) const {
        return (valid_mask[set] & (1u << way)) != 0;
    }

    inline bool is_dirty(int set, int way) const {
        return (dirty_mask[set] & (1u << way)) != 0;
    }

    inline void set_dirty(int set, int way, bool d) {
        if (d) dirty_mask[set] |= static_cast<uint8_t>(1u << way);
        else   dirty_mask[set] &= static_cast<uint8_t>(~(1u << way));
    }

    inline void set_valid(int set, int way, bool v) {
        if (v) valid_mask[set] |= static_cast<uint8_t>(1u << way);
        else   valid_mask[set] &= static_cast<uint8_t>(~(1u << way));
    }

    inline uint64_t get_tag(int set, int way) const {
        return tag[static_cast<size_t>(set) * WAYS + way];
    }

    inline void set_tag(int set, int way, uint64_t t) {
        tag[static_cast<size_t>(set) * WAYS + way] = t;
    }

    // Returns an invalid way if one exists, else the LRU victim way.
    inline int victim_way(int set) const {
        const uint8_t vmask = valid_mask[set];
        if (vmask != 0xFFu) {
            for (int w = 0; w < WAYS; ++w) {
                if (!(vmask & (1u << w))) return w;
            }
        }
        return lru_victim(lru[set]);
    }

    // Place {valid=true, dirty=d, tag=t} into `way` of `set` and mark MRU.
    inline void place_and_touch(int set, int way, bool d, uint64_t t) {
        set_valid(set, way, true);
        set_dirty(set, way, d);
        set_tag(set, way, t);
        touch(set, way);
    }
};

// ---------------------------------------------------------------------------
// CacheSim implementation
// ---------------------------------------------------------------------------
class FlatCacheSim final : public csot::CacheSim {
public:
    void on_init() override {
        l1_.init();
        l2_.init();
    }

    CacheStats run(const MemAccess* acc, std::size_t n) override {
        CacheStats s{};

        CacheLevel<L1_SETS>& L1 = l1_;
        CacheLevel<L2_SETS>& L2 = l2_;

        uint64_t reads = 0, writes = 0;
        uint64_t l1_hits = 0, l1_misses = 0;
        uint64_t l2_hits = 0, l2_misses = 0;
        uint64_t dirty_writebacks = 0;

        for (std::size_t i = 0; i < n; ++i) {
            const uint64_t addr = acc[i].address;
            const bool     wr   = acc[i].is_write != 0;

            if (wr) ++writes; else ++reads;

            const uint64_t b  = addr >> 6;
            const int      s1 = static_cast<int>(b & L1_SET_MASK);
            const uint64_t t1 = b >> L1_INDEX_BITS;

            // Prefetch the L1 set for the *next* access while we work on this one.
            if (i + 1 < n) {
                const uint64_t next_b  = acc[i + 1].address >> 6;
                const int      next_s1 = static_cast<int>(next_b & L1_SET_MASK);
                __builtin_prefetch(&L1.tag[static_cast<size_t>(next_s1) * WAYS], 0, 1);
            }

            const int w1 = L1.find(s1, t1);
            if (w1 >= 0) {
                ++l1_hits;
                L1.touch(s1, w1);
                if (wr) L1.set_dirty(s1, w1, true);
                continue;
            }
            ++l1_misses;

            const int      s2 = static_cast<int>(b & L2_SET_MASK);
            const uint64_t t2 = b >> L2_INDEX_BITS;

            const int w2 = L2.find(s2, t2);
            if (w2 >= 0) {
                ++l2_hits;
                L2.touch(s2, w2);
            } else {
                ++l2_misses;
                const int v = L2.victim_way(s2);
                if (L2.is_valid(s2, v) && L2.is_dirty(s2, v)) ++dirty_writebacks;
                L2.place_and_touch(s2, v, /*dirty=*/false, t2);
            }

            // ---- fill into L1 (write-allocate) ----
            const int v1 = L1.victim_way(s1);
            if (L1.is_valid(s1, v1) && L1.is_dirty(s1, v1)) {
                // Evict dirty L1 victim -> write back to L2.
                const uint64_t bv  = (L1.get_tag(s1, v1) << L1_INDEX_BITS) | static_cast<uint64_t>(s1);
                const int      s2v = static_cast<int>(bv & L2_SET_MASK);
                const uint64_t t2v = bv >> L2_INDEX_BITS;

                const int wv = L2.find(s2v, t2v);
                if (wv >= 0) {
                    L2.set_dirty(s2v, wv, true);  // not counted, not touched
                } else {
                    const int vv = L2.victim_way(s2v);
                    if (L2.is_valid(s2v, vv) && L2.is_dirty(s2v, vv)) ++dirty_writebacks;
                    L2.place_and_touch(s2v, vv, /*dirty=*/true, t2v);
                }
            }
            L1.place_and_touch(s1, v1, /*dirty=*/wr, t1);
        }

        s.reads            = reads;
        s.writes           = writes;
        s.l1_hits          = l1_hits;
        s.l1_misses        = l1_misses;
        s.l2_hits          = l2_hits;
        s.l2_misses        = l2_misses;
        s.dirty_writebacks = dirty_writebacks;
        return s;
    }

private:
    CacheLevel<L1_SETS> l1_;
    CacheLevel<L2_SETS> l2_;
};

}  // namespace

extern "C" csot::CacheSim* create_cache_sim() {
    return new FlatCacheSim();
}
