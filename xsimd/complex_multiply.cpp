#include <nanobench.h>

#include <array>
#include <complex>
#include <iostream>
#include <random>
#include <vector>
#include <xsimd/xsimd.hpp>

namespace xs = xsimd;

using vector_t = std::vector<std::complex<double>, xsimd::aligned_allocator<std::complex<double>, 64>>;

void benchmark_scalar(std::size_t N, const vector_t& a, const vector_t& b, vector_t& out,
                      ankerl::nanobench::Bench& bench) {
    bench.run("scalar complex mul N=" + std::to_string(N), [&] {
#pragma GCC ivdep
        for (std::size_t i = 0; i < N; ++i) {
            out[i] = a[i] * b[i];
        }
        ankerl::nanobench::doNotOptimizeAway(out);
    });
}

void benchmark_xsimd(std::size_t N, const vector_t& a, const vector_t& b, vector_t& out,
                     ankerl::nanobench::Bench& bench) {
    using batch_type = xs::batch<std::complex<double>>;
    constexpr std::size_t simd_size = batch_type::size;
    bench.run("xsimd complex mul N=" + std::to_string(N), [&] {
#pragma GCC ivdep
        for (std::size_t i = 0; i < N; i += simd_size) {
            batch_type va = batch_type::load_aligned(a.data() + i);
            batch_type vb = batch_type::load_aligned(b.data() + i);
            batch_type vc = va * vb;
            vc.store_aligned(out.data() + i);
        }
        ankerl::nanobench::doNotOptimizeAway(out);
    });
}

void benchmark_manual(std::size_t N, vector_t& a, vector_t& b, vector_t& out, ankerl::nanobench::Bench& bench) {
    using dbl_batch = xsimd::batch<double>;

    struct even_idx {
        static constexpr unsigned get(unsigned index, unsigned /*size*/) noexcept {
            // flip the low bit: 0→1, 1→0, 2→3, 3→2, …
            return index * 2;
        }
    };

    struct odd_idx {
        static constexpr unsigned get(unsigned index, unsigned /*size*/) noexcept {
            // flip the low bit: 0→1, 1→0, 2→3, 3→2, …
            return index * 2 + 1;
        }
    };
    const auto idx_re =
        xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, dbl_batch::arch_type, even_idx>().as_batch();
    const auto idx_im =
        xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, dbl_batch::arch_type, odd_idx>().as_batch();

    bench.run("xsimd complex mul (gather/scatter) N=" + std::to_string(N), [&] {
#pragma GCC ivdep
        for (std::size_t i = 0; i < 2 * N; i += dbl_batch::size * 2) {
            // load S/2 reals and S/2 imags without duplication
            const auto a_re = dbl_batch::gather(reinterpret_cast<const double*>(a.data()) + i, idx_re);
            const auto a_im = dbl_batch::gather(reinterpret_cast<const double*>(a.data()) + i, idx_im);
            const auto b_re = dbl_batch::gather(reinterpret_cast<const double*>(b.data()) + i, idx_re);
            const auto b_im = dbl_batch::gather(reinterpret_cast<const double*>(b.data()) + i, idx_im);

            // do the complex multiply on these half-width vectors
            const auto real = xsimd::fnma(a_im, b_im, a_re * b_re);
            const auto imag = xsimd::fma(a_re, b_im, a_im * b_re);
            // scatter back into interleaved [r0,i0,r1,i1,…] layout
            real.scatter(reinterpret_cast<double*>(out.data()) + i, idx_re);
            imag.scatter(reinterpret_cast<double*>(out.data()) + i, idx_im);
        }
        ankerl::nanobench::doNotOptimizeAway(out);
    });
}

void benchmark_swizzle(std::size_t N, vector_t& a, vector_t& b, vector_t& out, ankerl::nanobench::Bench& bench) {
    using batch_t = xsimd::batch<double>;
    using arch = batch_t::arch_type;
    constexpr std::size_t S = batch_t::size;  // # doubles per batch

    // — compile-time indexers for swizzle —
    struct real_dup {
        static constexpr unsigned get(const unsigned i, unsigned) noexcept { return i & ~1u; }
    };
    struct imag_dup {
        static constexpr unsigned get(const unsigned i, unsigned) noexcept { return i | 1u; }
    };
    struct swap_pair {
        static constexpr unsigned get(const unsigned i, unsigned) noexcept { return i ^ 1u; }
    };
    struct even_lane {
        static constexpr bool get(const unsigned i, unsigned) noexcept { return i & 1u; }
    };

    constexpr auto mask = xsimd::make_batch_bool_constant<double, arch, even_lane>();

    bench.run("xsimd complex mul (swizzle/select) N=" + std::to_string(N), [&] {
        // total doubles = 2*N
#pragma GCC ivdep
        for (std::size_t i = 0; i < 2 * N; i += S) {
            // load S interleaved doubles (= S/2 complexes)
            const auto va = batch_t::load_aligned(reinterpret_cast<const double*>(a.data()) + i);
            const auto vb = batch_t::load_aligned(reinterpret_cast<const double*>(b.data()) + i);

            // extract:
            const auto a_re = xsimd::swizzle(
                va,
                xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, arch, real_dup>());  // [r0,r0,r1,r1…]
            const auto a_im = xsimd::swizzle(
                va,
                xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, arch, imag_dup>());  // [i0,i0,i1,i1…]
            const auto b_sw =
                xsimd::swizzle(vb, xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, arch,
                                                              swap_pair>());  // [b1_re,b0_re,b3_re,b2_re…]
            // odd=(a_im * swapped-b), prod=(a_re * b)
            const auto odd = a_im * b_sw;

            const auto res = xsimd::fma(a_re, vb, xsimd::select(mask, odd, xsimd::neg(odd)));

            // store back interleaved
            res.store_aligned(reinterpret_cast<double*>(out.data()) + i);
        }
        ankerl::nanobench::doNotOptimizeAway(out);
    });
}

void benchmark_fma_add_sub(std::size_t N, vector_t& a, vector_t& b, vector_t& out, ankerl::nanobench::Bench& bench) {
    using pack      = xsimd::batch<double>;                  // 4 doubles on AVX2
    using arch      = pack::arch_type;
    constexpr std::size_t S = pack::size;                    // 4

    // --- compile-time helpers -------------------------------------------------
    struct swap_pair { static constexpr unsigned get(unsigned i, unsigned) noexcept { return i ^ 1u; } };
    struct dup_real  { static constexpr unsigned get(unsigned i, unsigned) noexcept { return i & ~1u; } };
    struct dup_imag  { static constexpr unsigned get(unsigned i, unsigned) noexcept { return i |  1u; } };
    struct odd_lane  { static constexpr bool     get(unsigned i, unsigned) noexcept { return i &  1u; } };

    constexpr auto swap_idx  = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, arch, swap_pair>();
    constexpr auto real_idx  = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, arch, dup_real>();
    constexpr auto imag_idx  = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, arch, dup_imag>();
    constexpr auto imag_mask = xsimd::make_batch_bool_constant<double, arch, odd_lane>();

    bench.run("xsimd complex mul (2 FMA + blend) N=" + std::to_string(N), [&] {
#pragma GCC ivdep
        for (std::size_t i = 0; i < 2 * N; i += S) {

            // 1. load [re0,im0,re1,im1]
            pack va = pack::load_aligned(reinterpret_cast<const double*>(a.data()) + i);
            pack vb = pack::load_aligned(reinterpret_cast<const double*>(b.data()) + i);

            // 2. duplicate real & imag parts of b
            pack vb_re = xsimd::swizzle(vb, real_idx);        // [br0,br0,br1,br1]
            pack vb_im = xsimd::swizzle(vb, imag_idx);        // [bi0,bi0,bi1,bi1]


            // 3. cross = (ai * bi, ar * bi, …)   using one swizzle on a
            pack va_sw = xsimd::swizzle(va, swap_idx);
            // 4. two FMAs: prod ± cross
            // pack real_lane = xsimd::fms(va, vb_re, va_sw * vb_im); // (ar*br) – (ai*bi)
            // pack imag_lane = xsimd::fma(va, vb_re, va_sw * vb_im); // (ar*br) + (ai*bi)

            // 5. blend even→real, odd→imag   [r0,i0,r1,i1]
            // pack result = xsimd::select(imag_mask, imag_lane, real_lane);
            // cross term stays pure-xsimd
            pack cross = va_sw * vb_im;                     // (ai*bi , ar*bi , …)

            // bit-cast the 3 operands to __m256d
            __m256d va_d     = xsimd::bit_cast<__m256d>(va);
            __m256d vb_re_d  = xsimd::bit_cast<__m256d>(vb_re);
            __m256d cross_d  = xsimd::bit_cast<__m256d>(cross);

            // even lanes: va*vb_re − cross → real
            // odd  lanes: va*vb_re + cross → imag
            __m256d res_d = _mm256_fmaddsub_pd(vb_re_d, va_d, cross_d);

            pack result = xsimd::bit_cast<pack>(res_d);
            result.store_aligned(reinterpret_cast<double*>(out.data()) + i);
        }
        ankerl::nanobench::doNotOptimizeAway(out);
    });
}

// assume vector_t is std::vector<std::complex<double>> (or equivalent)
void benchmark_intrinsics(std::size_t N, vector_t& a, vector_t& b, vector_t& out, ankerl::nanobench::Bench& bench) {
    bench.run("intrinsics complex mul N=" + std::to_string(N), [&] {
        const double* ap = reinterpret_cast<const double*>(a.data());
        const double* bp = reinterpret_cast<const double*>(b.data());
        double* op = reinterpret_cast<double*>(out.data());
        const std::size_t D = N * 2;  // total doubles

        std::size_t i = 0;
#pragma GCC ivdep
        for (; i + 3 < D; i += 4) {               // 4 doubles = 2 complex<double>
            __m256d va = _mm256_load_pd(ap + i);  // [ar0 ai0 ar1 ai1]
            __m256d vb = _mm256_load_pd(bp + i);  // [br0 bi0 br1 bi1]

            __m256d vb_re = _mm256_permute_pd(vb, 0x0);  // duplicate re parts
            __m256d vb_im = _mm256_permute_pd(vb, 0xF);  // duplicate im parts
            __m256d va_sw = _mm256_permute_pd(va, 0x5);  // swap   ar↔ai

            __m256d cross = _mm256_mul_pd(va_sw, vb_im);  // ai*bi / ar*bi
            __m256d result = _mm256_fmaddsub_pd(vb_re, va, cross);
            //  even lanes: vb_re*va - cross  → real
            //  odd  lanes: vb_re*va + cross  → imag

            _mm256_store_pd(op + i, result);
        }
        ankerl::nanobench::doNotOptimizeAway(out);
    });
}

int main() {
    volatile std::default_random_engine::result_type seed = 42;
    std::default_random_engine rng{seed};
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    ankerl::nanobench::Bench bench;
    bench.unit("mul").minEpochIterations(20000);
    constexpr bool print = false;  // set to true to print results
    for (std::size_t N = 8; N <= 16384; N *= 2) {
        bench.batch(N);
        vector_t a(N), b(N), out(N);

        for (std::size_t i = 0; i < N; ++i) {
            a[i] = {dist(rng), dist(rng)};
            b[i] = {dist(rng), dist(rng)};
        }

        benchmark_scalar(N, a, b, out, bench);
        if constexpr (print) {
            const auto sum = std::accumulate(out.begin(), out.end(), std::complex<double>(0.0, 0.0));
            std::cout << "N=" << N << " sum: (" << sum.real() << ", " << sum.imag() << ")" << std::endl;
        }
        benchmark_xsimd(N, a, b, out, bench);
        if constexpr (print) {
            const auto sum = std::accumulate(out.begin(), out.end(), std::complex<double>(0.0, 0.0));
            std::cout << "N=" << N << " sum: (" << sum.real() << ", " << sum.imag() << ")" << std::endl;
        }

        benchmark_manual(N, a, b, out, bench);
        if constexpr (print) {
            const auto sum = std::accumulate(out.begin(), out.end(), std::complex<double>(0.0, 0.0));
            std::cout << "N=" << N << " sum: (" << sum.real() << ", " << sum.imag() << ")" << std::endl;
        }
        benchmark_swizzle(N, a, b, out, bench);
        if constexpr (print) {
            const auto sum = std::accumulate(out.begin(), out.end(), std::complex<double>(0.0, 0.0));
            std::cout << "N=" << N << " sum: (" << sum.real() << ", " << sum.imag() << ")" << std::endl;
        }

        benchmark_fma_add_sub(N, a, b, out, bench);
        if constexpr (print) {
            const auto sum = std::accumulate(out.begin(), out.end(), std::complex<double>(0.0, 0.0));
            std::cout << "N=" << N << " sum: (" << sum.real() << ", " << sum.imag() << ")" << std::endl;
        }
        benchmark_intrinsics(N, a, b, out, bench);
        if constexpr (print) {
            const auto sum = std::accumulate(out.begin(), out.end(), std::complex<double>(0.0, 0.0));
            std::cout << "N=" << N << " sum: (" << sum.real() << ", " << sum.imag() << ")" << std::endl;
        }
    }

    return 0;
}