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

// Optimized compile-time swizzle for float (8 lanes)
// Optimized compile-time swizzle for float (8 lanes)
template <uint32_t V0, uint32_t V1, uint32_t V2, uint32_t V3, uint32_t V4, uint32_t V5, uint32_t V6, uint32_t V7>
static inline __m256 swizzle_const_opt(__m256 self) noexcept {
    // 1) identity?
    constexpr bool is_identity = (V0 == 0 && V1 == 1 && V2 == 2 && V3 == 3 && V4 == 4 && V5 == 5 && V6 == 6 && V7 == 7);

    // 2) all-different mask → full 8-lane permute
    constexpr uint32_t bitmask = (1u << (V0 & 7)) | (1u << (V1 & 7)) | (1u << (V2 & 7)) | (1u << (V3 & 7)) |
                                 (1u << (V4 & 7)) | (1u << (V5 & 7)) | (1u << (V6 & 7)) | (1u << (V7 & 7));
    constexpr bool is_all_different = (bitmask == 0xFFu);

    // 3) duplicate-low half?
    constexpr bool is_dup_lo = ((V0 < 4 && V1 < 4 && V2 < 4 && V3 < 4) && V4 == V0 && V5 == V1 && V6 == V2 && V7 == V3);

    // 4) duplicate-high half?
    constexpr bool is_dup_hi = (V0 >= 4 && V0 <= 7 && V1 >= 4 && V1 <= 7 && V2 >= 4 && V2 <= 7 && V3 >= 4 && V3 <= 7 &&
                                V4 == V0 && V5 == V1 && V6 == V2 && V7 == V3);

    XSIMD_IF_CONSTEXPR(is_identity) { return self; }
    else XSIMD_IF_CONSTEXPR(is_all_different) {
        // one-shot 8-lane permute
        const __m256i idx = _mm256_setr_epi32(V0, V1, V2, V3, V4, V5, V6, V7);
        return _mm256_permutevar8x32_ps(self, idx);
    }
    else XSIMD_IF_CONSTEXPR(is_dup_lo) {
        __m128 lo = _mm256_castps256_ps128(self);
        // if lo is not identity, we can permute it before duplicating
        XSIMD_IF_CONSTEXPR(V0 != 0 || V1 != 1 || V2 != 2 || V3 != 3) {
            constexpr int imm = ((V3 & 3) << 6) | ((V2 & 3) << 4) | ((V1 & 3) << 2) | ((V0 & 3) << 0);
            lo = _mm_permute_ps(lo, imm);
        }
        return _mm256_set_m128(lo, lo);
    }
    else XSIMD_IF_CONSTEXPR(is_dup_hi) {
        __m128 hi = _mm256_extractf128_ps(self, 1);
        XSIMD_IF_CONSTEXPR(V0 != 4 || V1 != 5 || V2 != 6 || V3 != 7) {
            constexpr int imm = ((V3 & 3) << 6) | ((V2 & 3) << 4) | ((V1 & 3) << 2) | ((V0 & 3) << 0);
            hi = _mm_permute_ps(hi, imm);
        }
        return _mm256_set_m128(hi, hi);
    }
    else {
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

// Optimized compile‐time swizzle for double (4 lanes)
template <uint64_t V0, uint64_t V1, uint64_t V2, uint64_t V3>
static inline __m256d swizzle_const_opt(__m256d self) noexcept {
    constexpr bool is_identity = (V0 == 0 && V1 == 1 && V2 == 2 && V3 == 3);
    constexpr bool can_use_pd = (V0 < 2 && V1 < 2 && V2 >= 2 && V2 < 4 && V3 >= 2 && V3 < 4);

    XSIMD_IF_CONSTEXPR(is_identity) { return self; }
    XSIMD_IF_CONSTEXPR(can_use_pd) {
        // build the 4-bit immediate: bit i = 1 if you pick the upper element of pair i
        constexpr int mask = ((V0 & 1) << 0) | ((V1 & 1) << 1) | ((V2 & 1) << 2) | ((V3 & 1) << 3);
        return _mm256_permute_pd(self, mask);
    }
    // fallback to full 4-element permute
    constexpr int imm = ((V3 & 3) << 6) | ((V2 & 3) << 4) | ((V1 & 3) << 2) | ((V0 & 3) << 0);
    return _mm256_permute4x64_pd(self, imm);
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
    if (std::memcmp(&a, &b, sizeof(a)) != 0) {
        std::cerr << "Mismatch: " << a << " vs " << b << std::endl;
        return false;
    }
    return true;
}
template <typename T, int N>
void print_array(const T (&arr)[N], const char* label) {
    std::cout << label << " ";
    std::cout << "[";
    for (int i = 0; i < N; ++i) {
        std::cout << arr[i] << (i + 1 < N ? ", " : "");
    }
    std::cout << "]\n";
}
void check4d(const char* name, __m256d orig, __m256d opt) {
    double a[4], b[4];
    store4d(a, orig);
    store4d(b, opt);
    std::cout << name << ":\n";
    print_array(a, "  orig:");
    print_array(b, "   opt:");
    for (int i = 0; i < 4; ++i) {
        assert(almost_equal(a[i], b[i]));
    }
    std::cout << "\n";
}

void check8f(const char* name, __m256 orig, __m256 opt) {
    float a[8], b[8];
    store8f(a, orig);
    store8f(b, opt);
    std::cout << name << ":\n";
    print_array(a, "  orig:");
    print_array(b, "   opt:");
    for (int i = 0; i < 8; ++i) {
        assert(almost_equal(a[i], b[i]));
    }
    std::cout << "\n";
}
// --------------------------------------------------------
// Revised test_correctness()
// --------------------------------------------------------
void test_correctness() {
    alignas(32) float in8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    alignas(32) double in4[4] = {0, 1, 2, 3};

    // print originals once
    print_array(in8, "Input float vector : ");
    std::cout << "\n";
    print_array(in4, "Input double vector: ");
    std::cout << "\n\n";

    __m256 v8 = load8f(in8);
    __m256d v4 = load4d(in4);

    // CT‐float identity & reverse
    check8f("CT-float identity", swizzle_const_orig<0, 1, 2, 3, 4, 5, 6, 7>(v8),
            swizzle_const_opt<0, 1, 2, 3, 4, 5, 6, 7>(v8));
    check8f("CT-float reverse", swizzle_const_orig<7, 6, 5, 4, 3, 2, 1, 0>(v8),
            swizzle_const_opt<7, 6, 5, 4, 3, 2, 1, 0>(v8));
    std::cout << "\n";

    // CT‐double identity & reverse
    check4d("CT-double identity", swizzle_const_orig<0, 1, 2, 3>(v4), swizzle_const_opt<0, 1, 2, 3>(v4));
    check4d("CT-double reverse", swizzle_const_orig<3, 2, 1, 0>(v4), swizzle_const_opt<3, 2, 1, 0>(v4));
    std::cout << "\n";

    // RT‐float identity & reverse
    alignas(32) int32_t m8_rev[8] = {7, 6, 5, 4, 3, 2, 1, 0};
    __m256i mi8_rev = _mm256_load_si256((__m256i*)m8_rev);
    check8f("RT-float reverse", swizzle_var_orig(v8, mi8_rev), swizzle_var_opt(v8, mi8_rev));
    alignas(32) int32_t m8_id[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    __m256i mi8_id = _mm256_load_si256((__m256i*)m8_id);
    check8f("RT-float identity", swizzle_var_orig(v8, mi8_id), swizzle_var_opt(v8, mi8_id));
    std::cout << "\n";

    // RT‐double identity & reverse
    alignas(32) int64_t m4_rev[4] = {3, 2, 1, 0};
    __m256i mi4_rev = _mm256_load_si256((__m256i*)m4_rev);
    check4d("RT-double reverse", swizzle_var_orig(v4, mi4_rev), swizzle_var_opt(v4, mi4_rev));
    alignas(32) int64_t m4_id[4] = {0, 1, 2, 3};
    __m256i mi4_id = _mm256_load_si256((__m256i*)m4_id);
    check4d("RT-double identity", swizzle_var_orig(v4, mi4_id), swizzle_var_opt(v4, mi4_id));
}

void test_special_pd_cases() {
    alignas(32) double data[4] = {1.0, 2.0, 3.0, 4.0};
    print_array(data, "Input double vector:");
    __m256d v = load4d(data);

    check4d("duplicate real parts", swizzle_const_orig<0, 0, 2, 2>(v), swizzle_const_opt<0, 0, 2, 2>(v));

    check4d("duplicate imag parts", swizzle_const_orig<1, 1, 3, 3>(v), swizzle_const_opt<1, 1, 3, 3>(v));

    check4d("swap real<->imag", swizzle_const_orig<1, 0, 3, 2>(v), swizzle_const_opt<1, 0, 3, 2>(v));

    check4d("identity", swizzle_const_orig<0, 1, 2, 3>(v), swizzle_const_opt<0, 1, 2, 3>(v));

    check4d("dup low pair", swizzle_const_orig<0, 0, 1, 1>(v), swizzle_const_opt<0, 0, 1, 1>(v));

    check4d("pairwise dup <2,2,3,3>", swizzle_const_orig<2, 2, 3, 3>(v), swizzle_const_opt<2, 2, 3, 3>(v));

    check4d("generic fallback <0,2,2,0>", swizzle_const_orig<0, 2, 2, 0>(v), swizzle_const_opt<0, 2, 2, 0>(v));

    std::cout << "special PD cases passed\n\n";
}

void test_special_ps_cases() {
    alignas(32) float in[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    print_array(in, "Input float vector:");
    __m256 v = load8f(in);

    check8f("identity", swizzle_const_orig<0, 1, 2, 3, 4, 5, 6, 7>(v), swizzle_const_opt<0, 1, 2, 3, 4, 5, 6, 7>(v));

    check8f("full reverse", swizzle_const_orig<3, 2, 1, 0, 7, 6, 5, 4>(v),
            swizzle_const_opt<3, 2, 1, 0, 7, 6, 5, 4>(v));

    check8f("duplicate low half", swizzle_const_orig<0, 1, 2, 3, 0, 1, 2, 3>(v),
            swizzle_const_opt<0, 1, 2, 3, 0, 1, 2, 3>(v));

    check8f("duplicate high half", swizzle_const_orig<4, 5, 6, 7, 4, 5, 6, 7>(v),
            swizzle_const_opt<4, 5, 6, 7, 4, 5, 6, 7>(v));

    check8f("duplicate and permute low half", swizzle_const_orig<2, 1, 0, 3, 2, 1, 0, 3>(v),
            swizzle_const_opt<2, 1, 0, 3, 2, 1, 0, 3>(v));

    check8f("duplicate and permute high half", swizzle_const_orig<7, 5, 6, 4, 7, 5, 6, 4>(v),
            swizzle_const_opt<7, 5, 6, 4, 7, 5, 6, 4>(v));

    check8f("pairwise dup <0,0,1,1,2,2,3,3>", swizzle_const_orig<0, 0, 1, 1, 2, 2, 3, 3>(v),
            swizzle_const_opt<0, 0, 1, 1, 2, 2, 3, 3>(v));

    check8f("pairwise dup <4,4,5,5,6,6,7,7>", swizzle_const_orig<4, 4, 5, 5, 6, 6, 7, 7>(v),
            swizzle_const_opt<4, 4, 5, 5, 6, 6, 7, 7>(v));

    std::cout << "special PS cases passed\n\n";
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
