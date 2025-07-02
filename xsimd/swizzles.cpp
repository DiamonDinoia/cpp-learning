// File: swizzles_optimized_test.cpp
#include <immintrin.h>
#include <nanobench.h>

#include <cassert>
#include <cstring>  // for memcmp
#include <iostream>
#include <random>
#include <xsimd/xsimd.hpp>

template <uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3, uint32_t V4, uint32_t V5, uint32_t V6, uint32_t V7>
static inline __m256 swizzle_const_orig(__m256 self) noexcept {
    __m256 hi = _mm256_castps128_ps256(_mm256_extractf128_ps(self, 1));
    __m256 hi_hi = _mm256_insertf128_ps(self, _mm256_castps256_ps128(hi), 0);
    __m256 low = _mm256_castps128_ps256(_mm256_castps256_ps128(self));
    __m256 low_lo = _mm256_insertf128_ps(self, _mm256_castps256_ps128(low), 1);

    constexpr int idx[8] = {int(V0 % 4), int(V1 % 4), int(V2 % 4), int(V3 % 4),
                            int(V4 % 4), int(V5 % 4), int(V6 % 4), int(V7 % 4)};
    __m256i ctrl = _mm256_setr_epi32(idx[0], idx[1], idx[2], idx[3], idx[4], idx[5], idx[6], idx[7]);

    __m256 r0 = _mm256_permutevar_ps(low_lo, ctrl);
    __m256 r1 = _mm256_permutevar_ps(hi_hi, ctrl);

    constexpr int blend_mask = ((V0 >= 4) << 0) | ((V1 >= 4) << 1) | ((V2 >= 4) << 2) | ((V3 >= 4) << 3) |
                               ((V4 >= 4) << 4) | ((V5 >= 4) << 5) | ((V6 >= 4) << 6) | ((V7 >= 4) << 7);

    return _mm256_blend_ps(r0, r1, blend_mask);
}

// Optimized
template <uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3, uint32_t V4, uint32_t V5, uint32_t V6, uint32_t V7>
static inline __m256 swizzle_const_opt(__m256 self) noexcept {
    constexpr bool is_identity = (V0 == 0 && V1 == 1 && V2 == 2 && V3 == 3 && V4 == 4 && V5 == 5 && V6 == 6 && V7 == 7);
    constexpr bool is_reverse = (V0 == 3 && V1 == 2 && V2 == 1 && V3 == 0 && V4 == 7 && V5 == 6 && V6 == 5 && V7 == 4);
    constexpr bool is_dup_lo = (V0 == 0 && V1 == 1 && V2 == 2 && V3 == 3 && V4 == 0 && V5 == 1 && V6 == 2 && V7 == 3);
    constexpr bool is_dup_hi = (V0 == 4 && V1 == 5 && V2 == 6 && V3 == 7 && V4 == 4 && V5 == 5 && V6 == 6 && V7 == 7);
    constexpr bool is_pairdup_lo = (V0 == 0 && V1 == 0 && V2 == 1 && V3 == 1);
    constexpr bool is_pairdup_hi = (V4 == 2 && V5 == 2 && V6 == 3 && V7 == 3);
    constexpr bool is_pairdup = is_pairdup_lo && is_pairdup_hi;

    XSIMD_IF_CONSTEXPR (is_identity) {
        return self;
    } else XSIMD_IF_CONSTEXPR (is_reverse) {
        __m128 lo = _mm256_castps256_ps128(self);
        __m128 hi = _mm256_extractf128_ps(self, 1);
        __m128 lo_rev = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(0, 1, 2, 3));
        __m128 hi_rev = _mm_shuffle_ps(hi, hi, _MM_SHUFFLE(0, 1, 2, 3));
        return _mm256_set_m128(lo_rev, hi_rev);
    } else XSIMD_IF_CONSTEXPR (is_dup_lo) {
        __m128 lo = _mm256_castps256_ps128(self);
        return _mm256_set_m128(lo, lo);
    } else XSIMD_IF_CONSTEXPR (is_dup_hi) {
        __m128 hi = _mm256_extractf128_ps(self, 1);
        return _mm256_set_m128(hi, hi);
    } else XSIMD_IF_CONSTEXPR (is_pairdup) {
        __m256i idx = _mm256_setr_epi32(V0, V0, V2, V2, V4, V4, V6, V6);
        return _mm256_permutevar8x32_ps(self, idx);
    } else {
        __m128 lo = _mm256_castps256_ps128(self);
        __m128 hi = _mm256_extractf128_ps(self, 1);

        constexpr int lo_im = _MM_SHUFFLE(int(V3 % 4), int(V2 % 4), int(V1 % 4), int(V0 % 4));
        constexpr int hi_im = _MM_SHUFFLE(int(V7 % 4), int(V6 % 4), int(V5 % 4), int(V4 % 4));

        __m128 lo_s = _mm_shuffle_ps(lo, lo, lo_im);
        __m128 hi_s = _mm_shuffle_ps(hi, hi, hi_im);

        return _mm256_set_m128(hi_s, lo_s);
    }
}

// --- Compile-time mask, double (4 lanes) ---

// Original
template <uint64_t V0, uint64_t V1, uint64_t V2, uint64_t V3>
static inline __m256d swizzle_const_orig(__m256d self) noexcept {
    __m256d hi = _mm256_castpd128_pd256(_mm256_extractf128_pd(self, 1));
    __m256d hi_hi = _mm256_insertf128_pd(self, _mm256_castpd256_pd128(hi), 0);
    __m256d low = _mm256_castpd128_pd256(_mm256_castpd256_pd128(self));
    __m256d low_lo = _mm256_insertf128_pd(self, _mm256_castpd256_pd128(low), 1);

    constexpr long long idx[4] = {(V0 % 2) ? -1LL : 0LL, (V1 % 2) ? -1LL : 0LL, (V2 % 2) ? -1LL : 0LL,
                                  (V3 % 2) ? -1LL : 0LL};
    __m256i ctrl = _mm256_setr_epi64x(idx[0], idx[1], idx[2], idx[3]);

    __m256d r0 = _mm256_permutevar_pd(low_lo, ctrl);
    __m256d r1 = _mm256_permutevar_pd(hi_hi, ctrl);

    constexpr int blend_mask = ((V0 >= 2) << 0) | ((V1 >= 2) << 1) | ((V2 >= 2) << 2) | ((V3 >= 2) << 3);

    return _mm256_blend_pd(r0, r1, blend_mask);
}

// Optimized
template <uint64_t V0, uint64_t V1, uint64_t V2, uint64_t V3>
static inline __m256d swizzle_const_opt(__m256d self) noexcept {
    constexpr bool is_dup_re = (V0 % 2 == 0 && V1 % 2 == 0 && V2 % 2 == 0 && V3 % 2 == 0);
    constexpr bool is_dup_im = (V0 % 2 == 1 && V1 % 2 == 1 && V2 % 2 == 1 && V3 % 2 == 1);
    constexpr bool is_swap = (V0 % 2 == 1 && V1 % 2 == 0 && V2 % 2 == 1 && V3 % 2 == 0);
    constexpr bool is_identity = (V0 == 0 && V1 == 1 && V2 == 2 && V3 == 3);
    constexpr bool is_pairdup = (V0 == V1 && V2 == V3);

    XSIMD_IF_CONSTEXPR (is_identity) {
        return self;
    } else XSIMD_IF_CONSTEXPR (is_dup_re) {
        return _mm256_permute_pd(self, 0x0);
    } else XSIMD_IF_CONSTEXPR (is_dup_im) {
        return _mm256_permute_pd(self, 0xF);
    } else XSIMD_IF_CONSTEXPR (is_swap) {
        return _mm256_permute_pd(self, 0x5);
    } else XSIMD_IF_CONSTEXPR (is_pairdup) {
        constexpr int permute_mask = ((V2 & 3) << 2) | (V0 & 3);
        return _mm256_permute4x64_pd(self, permute_mask);
    } else {
        __m128d lo = _mm256_castpd256_pd128(self);
        __m128d hi = _mm256_extractf128_pd(self, 1);

        constexpr int lo_ctrl = ((V0 % 2) << 0) | ((V1 % 2) << 1);
        constexpr int hi_ctrl = ((V2 % 2) << 0) | ((V3 % 2) << 1);

        __m128d lo_s = _mm_shuffle_pd(lo, lo, lo_ctrl);
        __m128d hi_s = _mm_shuffle_pd(hi, hi, hi_ctrl);

        return _mm256_set_m128d(hi_s, lo_s);
    }
}

// --- Run-time mask, float ---

// Original
static inline __m256 swizzle_var_orig(__m256 self, __m256i mask_i) noexcept {
    __m256 hi = _mm256_castps128_ps256(_mm256_extractf128_ps(self, 1));
    __m256 hi_hi = _mm256_insertf128_ps(self, _mm256_castps256_ps128(hi), 0);
    __m256 low = _mm256_castps128_ps256(_mm256_castps256_ps128(self));
    __m256 low_lo = _mm256_insertf128_ps(self, _mm256_castps256_ps128(low), 1);

    __m256i half_i = _mm256_and_si256(mask_i, _mm256_set1_epi32(3));
    __m256 r0 = _mm256_permutevar_ps(low_lo, half_i);
    __m256 r1 = _mm256_permutevar_ps(hi_hi, half_i);

    __m256i gt4_i = _mm256_cmpgt_epi32(mask_i, _mm256_set1_epi32(3));
    __m256 blend = _mm256_castsi256_ps(gt4_i);

    return _mm256_blendv_ps(r0, r1, blend);
}

// Optimized
static inline __m256 swizzle_var_opt(__m256 self, __m256i mask_i) noexcept {
    __m128 lo = _mm256_castps256_ps128(self);
    __m128 hi = _mm256_extractf128_ps(self, 1);
    __m256 b_lo = _mm256_set_m128(lo, lo);
    __m256 b_hi = _mm256_set_m128(hi, hi);

    __m256i half_i = _mm256_and_si256(mask_i, _mm256_set1_epi32(3));
    __m256 r0 = _mm256_permutevar_ps(b_lo, half_i);
    __m256 r1 = _mm256_permutevar_ps(b_hi, half_i);

    __m256i gt4_i = _mm256_cmpgt_epi32(mask_i, _mm256_set1_epi32(3));
    __m256 blend = _mm256_castsi256_ps(gt4_i);
    return _mm256_blendv_ps(r0, r1, blend);
}

// --- Run-time mask, double ---

// Original
static inline __m256d swizzle_var_orig(__m256d self, __m256i mask_i) noexcept {
    __m256d hi = _mm256_castpd128_pd256(_mm256_extractf128_pd(self, 1));
    __m256d hi_hi = _mm256_insertf128_pd(self, _mm256_castpd256_pd128(hi), 0);
    __m256d low = _mm256_castpd128_pd256(_mm256_castpd256_pd128(self));
    __m256d low_lo = _mm256_insertf128_pd(self, _mm256_castpd256_pd128(low), 1);

    __m256i bit0 = _mm256_and_si256(mask_i, _mm256_set1_epi64x(1));
    __m256i half_i = _mm256_sub_epi64(_mm256_setzero_si256(), bit0);

    __m256d r0 = _mm256_permutevar_pd(low_lo, half_i);
    __m256d r1 = _mm256_permutevar_pd(hi_hi, half_i);

    __m256i gt2_i = _mm256_cmpgt_epi64(mask_i, _mm256_set1_epi64x(1));
    __m256d blend = _mm256_castsi256_pd(gt2_i);

    return _mm256_blendv_pd(r0, r1, blend);
}

// Optimized
static inline __m256d swizzle_var_opt(__m256d self, __m256i mask_i) noexcept {
    __m128d lo = _mm256_castpd256_pd128(self);
    __m128d hi = _mm256_extractf128_pd(self, 1);
    __m256d b_lo = _mm256_set_m128d(lo, lo);
    __m256d b_hi = _mm256_set_m128d(hi, hi);

    __m256i bit0 = _mm256_and_si256(mask_i, _mm256_set1_epi64x(1));
    __m256i half_i = _mm256_sub_epi64(_mm256_setzero_si256(), bit0);

    __m256d r0 = _mm256_permutevar_pd(b_lo, half_i);
    __m256d r1 = _mm256_permutevar_pd(b_hi, half_i);

    __m256i gt2_i = _mm256_cmpgt_epi64(mask_i, _mm256_set1_epi64x(1));
    __m256d blend = _mm256_castsi256_pd(gt2_i);
    return _mm256_blendv_pd(r0, r1, blend);
}

// Helpers
inline __m256 load8f(const float* p) { return _mm256_loadu_ps(p); }
inline void store8f(float* p, __m256 v) { _mm256_storeu_ps(p, v); }
inline __m256d load4d(const double* p) { return _mm256_loadu_pd(p); }
inline void store4d(double* p, __m256d v) { _mm256_storeu_pd(p, v); }

// Almost‐equal via bit‐wise compare
template <typename T>
bool almost_equal(T a, T b) {
    return std::memcmp(&a, &b, sizeof(a)) == 0;
}

// Correctness check
void test_correctness() {
    alignas(32) float in8[8] = {0, 1, 2, 3, 4, 5, 6, 7}, out1[8], out2[8];
    __m256 v8 = load8f(in8);
    alignas(32) int32_t m8[8] = {7, 6, 5, 4, 3, 2, 1, 0};
    __m256i mi8 = _mm256_load_si256((__m256i*)m8);

    alignas(32) double in4[4] = {0, 1, 2, 3}, outd1[4], outd2[4];
    __m256d v4 = load4d(in4);
    alignas(32) int64_t m4[4] = {3, 2, 1, 0};
    __m256i mi4 = _mm256_load_si256((__m256i*)m4);

    // CT‐float identity & reverse
    {
        auto o0 = swizzle_const_orig<0, 1, 2, 3, 4, 5, 6, 7>(v8);
        auto o1 = swizzle_const_opt<0, 1, 2, 3, 4, 5, 6, 7>(v8);
        store8f(out1, o0);
        store8f(out2, o1);
        for (int i = 0; i < 8; ++i) assert(almost_equal(out1[i], out2[i]));

        o0 = swizzle_const_orig<7, 6, 5, 4, 3, 2, 1, 0>(v8);
        o1 = swizzle_const_opt<7, 6, 5, 4, 3, 2, 1, 0>(v8);
        store8f(out1, o0);
        store8f(out2, o1);
        for (int i = 0; i < 8; ++i) assert(almost_equal(out1[i], out2[i]));
    }

    // CT‐double identity & reverse
    {
        auto d0 = swizzle_const_orig<0, 1, 2, 3>(v4);
        auto d1 = swizzle_const_opt<0, 1, 2, 3>(v4);
        store4d(outd1, d0);
        store4d(outd2, d1);
        for (int i = 0; i < 4; ++i) assert(almost_equal(outd1[i], outd2[i]));

        d0 = swizzle_const_orig<3, 2, 1, 0>(v4);
        d1 = swizzle_const_opt<3, 2, 1, 0>(v4);
        store4d(outd1, d0);
        store4d(outd2, d1);
        for (int i = 0; i < 4; ++i) assert(almost_equal(outd1[i], outd2[i]));
    }

    // RT‐float identity & reverse
    {
        auto r0 = swizzle_var_orig(v8, mi8 /*reverse*/);
        auto r1 = swizzle_var_opt(v8, mi8);
        store8f(out1, r0);
        store8f(out2, r1);
        for (int i = 0; i < 8; ++i) assert(almost_equal(out1[i], out2[i]));

        // identity mask
        alignas(32) int32_t id8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
        __m256i idi = _mm256_load_si256((__m256i*)id8);
        r0 = swizzle_var_orig(v8, idi);
        r1 = swizzle_var_opt(v8, idi);
        store8f(out1, r0);
        store8f(out2, r1);
        for (int i = 0; i < 8; ++i) assert(almost_equal(out1[i], out2[i]));
    }

    // RT‐double identity & reverse
    {
        auto R0 = swizzle_var_orig(v4, mi4);
        auto R1 = swizzle_var_opt(v4, mi4);
        store4d(outd1, R0);
        store4d(outd2, R1);
        for (int i = 0; i < 4; ++i) assert(almost_equal(outd1[i], outd2[i]));

        alignas(32) int64_t id4[4] = {0, 1, 2, 3};
        __m256i idi = _mm256_load_si256((__m256i*)id4);
        R0 = swizzle_var_orig(v4, idi);
        R1 = swizzle_var_opt(v4, idi);
        store4d(outd1, R0);
        store4d(outd2, R1);
        for (int i = 0; i < 4; ++i) assert(almost_equal(outd1[i], outd2[i]));
    }

    std::cout << "✅ correctness OK\n";
}

void test_special_pd_cases() {
    alignas(32) double data[4] = {1.0, 2.0, 3.0, 4.0};
    __m256d v = load4d(data);
    double o0[4], o1[4];

    // duplicate real parts
    auto r0 = swizzle_const_orig<0, 0, 2, 2>(v);
    auto r1 = swizzle_const_opt<0, 0, 2, 2>(v);
    store4d(o0, r0);
    store4d(o1, r1);
    for (int i = 0; i < 4; ++i) assert(almost_equal(o0[i], o1[i]));

    // duplicate imag parts
    r0 = swizzle_const_orig<1, 1, 3, 3>(v);
    r1 = swizzle_const_opt<1, 1, 3, 3>(v);
    store4d(o0, r0);
    store4d(o1, r1);
    for (int i = 0; i < 4; ++i) assert(almost_equal(o0[i], o1[i]));

    // swap real <-> imag
    r0 = swizzle_const_orig<1, 0, 3, 2>(v);
    r1 = swizzle_const_opt<1, 0, 3, 2>(v);
    store4d(o0, r0);
    store4d(o1, r1);
    for (int i = 0; i < 4; ++i) assert(almost_equal(o0[i], o1[i]));

    // identity (baseline)
    r0 = swizzle_const_orig<0, 1, 2, 3>(v);
    r1 = swizzle_const_opt<0, 1, 2, 3>(v);
    store4d(o0, r0);
    store4d(o1, r1);

    for (int i = 0; i < 4; ++i) assert(almost_equal(o0[i], o1[i]));
    r0 = swizzle_const_orig<0, 0, 1, 1>(v);
    r1 = swizzle_const_opt<0, 0, 1, 1>(v);
    store4d(o0, r0);
    store4d(o1, r1);
    for (int i = 0; i < 4; ++i) assert(almost_equal(o0[i], o1[i]));

    // pairwise duplication: 2,2,3,3
    r0 = swizzle_const_orig<2, 2, 3, 3>(v);
    r1 = swizzle_const_opt<2, 2, 3, 3>(v);
    store4d(o0, r0);
    store4d(o1, r1);
    for (int i = 0; i < 4; ++i) assert(almost_equal(o0[i], o1[i]));
    std::cout << "✅ special PD cases passed\n";
}

void test_special_ps_cases() {
    alignas(32) float in[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    __m256 v = load8f(in);
    float o0[8], o1[8];

    // identity
    auto r0 = swizzle_const_orig<0, 1, 2, 3, 4, 5, 6, 7>(v);
    auto r1 = swizzle_const_opt<0, 1, 2, 3, 4, 5, 6, 7>(v);
    store8f(o0, r0);
    store8f(o1, r1);
    for (int i = 0; i < 8; ++i) assert(almost_equal(o0[i], o1[i]));

    // full reverse
    r0 = swizzle_const_orig<3, 2, 1, 0, 7, 6, 5, 4>(v);
    r1 = swizzle_const_opt<3, 2, 1, 0, 7, 6, 5, 4>(v);
    store8f(o0, r0);
    store8f(o1, r1);
    for (int i = 0; i < 8; ++i) assert(almost_equal(o0[i], o1[i]));

    // duplicate low
    r0 = swizzle_const_orig<0, 1, 2, 3, 0, 1, 2, 3>(v);
    r1 = swizzle_const_opt<0, 1, 2, 3, 0, 1, 2, 3>(v);
    store8f(o0, r0);
    store8f(o1, r1);
    for (int i = 0; i < 8; ++i) assert(almost_equal(o0[i], o1[i]));

    // duplicate high
    r0 = swizzle_const_orig<4, 5, 6, 7, 4, 5, 6, 7>(v);
    r1 = swizzle_const_opt<4, 5, 6, 7, 4, 5, 6, 7>(v);
    store8f(o0, r0);
    store8f(o1, r1);
    for (int i = 0; i < 8; ++i) assert(almost_equal(o0[i], o1[i]));

    r0 = swizzle_const_orig<0, 0, 1, 1, 2, 2, 3, 3>(v);
    r1 = swizzle_const_opt<0, 0, 1, 1, 2, 2, 3, 3>(v);
    store8f(o0, r0);
    store8f(o1, r1);
    for (int i = 0; i < 8; ++i) assert(almost_equal(o0[i], o1[i]));

    // 4,4,5,5,6,6,7,7
    r0 = swizzle_const_orig<4, 4, 5, 5, 6, 6, 7, 7>(v);
    r1 = swizzle_const_opt<4, 4, 5, 5, 6, 6, 7, 7>(v);
    store8f(o0, r0);
    store8f(o1, r1);
    for (int i = 0; i < 8; ++i) assert(almost_equal(o0[i], o1[i]));

    std::cout << "✅ special PS cases passed\n";
}

// Benchmark
void run_bench() {
    alignas(32) float in8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    volatile __m256 v8 = load8f(in8);
    volatile __m256i mi8_rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    volatile __m256i mi8_id = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

    alignas(32) double in4[4] = {0, 1, 2, 3};
    volatile __m256d v4 = load4d(in4);
    volatile __m256i mi4_rev = _mm256_setr_epi64x(3, 2, 1, 0);
    volatile __m256i mi4_id = _mm256_setr_epi64x(0, 1, 2, 3);

    ankerl::nanobench::Bench bench;
    bench.title("AVX2 swizzles").minEpochIterations(20000);

    // Compile-time float
    bench.run("float compile-time identity [orig]", [&] {
        auto r = swizzle_const_orig<0, 1, 2, 3, 4, 5, 6, 7>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("float compile-time identity [opt]", [&] {
        auto r = swizzle_const_opt<0, 1, 2, 3, 4, 5, 6, 7>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("float compile-time reverse [orig]", [&] {
        auto r = swizzle_const_orig<7, 6, 5, 4, 3, 2, 1, 0>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("float compile-time reverse [opt]", [&] {
        auto r = swizzle_const_opt<7, 6, 5, 4, 3, 2, 1, 0>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("float compile-time dup-low [orig]", [&] {
        auto r = swizzle_const_orig<0, 1, 2, 3, 0, 1, 2, 3>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("float compile-time dup-low [opt]", [&] {
        auto r = swizzle_const_opt<0, 1, 2, 3, 0, 1, 2, 3>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("float compile-time dup-high [orig]", [&] {
        auto r = swizzle_const_orig<4, 5, 6, 7, 4, 5, 6, 7>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("float compile-time dup-high [opt]", [&] {
        auto r = swizzle_const_opt<4, 5, 6, 7, 4, 5, 6, 7>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("float compile-time pairdup-all [orig]", [&] {
        auto r = swizzle_const_orig<0, 0, 1, 1, 2, 2, 3, 3>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("float compile-time pairdup-all [opt]", [&] {
        auto r = swizzle_const_opt<0, 0, 1, 1, 2, 2, 3, 3>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("float compile-time pairdup-high-only [orig]", [&] {
        auto r = swizzle_const_orig<4, 4, 5, 5, 6, 6, 7, 7>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("float compile-time pairdup-high-only [opt]", [&] {
        auto r = swizzle_const_opt<4, 4, 5, 5, 6, 6, 7, 7>(v8);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    // Compile-time double
    bench.run("double compile-time identity [orig]", [&] {
        auto r = swizzle_const_orig<0, 1, 2, 3>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("double compile-time identity [opt]", [&] {
        auto r = swizzle_const_opt<0, 1, 2, 3>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("double compile-time reverse [orig]", [&] {
        auto r = swizzle_const_orig<3, 2, 1, 0>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("double compile-time reverse [opt]", [&] {
        auto r = swizzle_const_opt<3, 2, 1, 0>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("double compile-time dup-re [orig]", [&] {
        auto r = swizzle_const_orig<0, 0, 2, 2>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("double compile-time dup-re [opt]", [&] {
        auto r = swizzle_const_opt<0, 0, 2, 2>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("double compile-time dup-im [orig]", [&] {
        auto r = swizzle_const_orig<1, 1, 3, 3>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("double compile-time dup-im [opt]", [&] {
        auto r = swizzle_const_opt<1, 1, 3, 3>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("double compile-time swap-re-im [orig]", [&] {
        auto r = swizzle_const_orig<1, 0, 3, 2>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("double compile-time swap-re-im [opt]", [&] {
        auto r = swizzle_const_opt<1, 0, 3, 2>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("double compile-time pairdup-low [orig]", [&] {
        auto r = swizzle_const_orig<0, 0, 1, 1>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("double compile-time pairdup-low [opt]", [&] {
        auto r = swizzle_const_opt<0, 0, 1, 1>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("double compile-time pairdup-high [orig]", [&] {
        auto r = swizzle_const_orig<2, 2, 3, 3>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("double compile-time pairdup-high [opt]", [&] {
        auto r = swizzle_const_opt<2, 2, 3, 3>(v4);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    // Run-time float
    bench.run("float run-time identity [orig]", [&] {
        auto r = swizzle_var_orig(v8, mi8_id);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("float run-time identity [opt]", [&] {
        auto r = swizzle_var_opt(v8, mi8_id);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("float run-time reverse [orig]", [&] {
        auto r = swizzle_var_orig(v8, mi8_rev);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("float run-time reverse [opt]", [&] {
        auto r = swizzle_var_opt(v8, mi8_rev);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    // Run-time double
    bench.run("double run-time identity [orig]", [&] {
        auto r = swizzle_var_orig(v4, mi4_id);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("double run-time identity [opt]", [&] {
        auto r = swizzle_var_opt(v4, mi4_id);
        ankerl::nanobench::doNotOptimizeAway(r);
    });

    bench.run("double run-time reverse [orig]", [&] {
        auto r = swizzle_var_orig(v4, mi4_rev);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
    bench.run("double run-time reverse [opt]", [&] {
        auto r = swizzle_var_opt(v4, mi4_rev);
        ankerl::nanobench::doNotOptimizeAway(r);
    });
}

int main() {
    test_correctness();
    test_special_pd_cases();
    test_special_ps_cases();
    run_bench();
    return 0;
}
