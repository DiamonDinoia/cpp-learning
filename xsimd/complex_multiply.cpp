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
        static constexpr unsigned get(unsigned index, unsigned /*size*/) noexcept { return index * 2; }
    };
    struct odd_idx {
        static constexpr unsigned get(unsigned index, unsigned /*size*/) noexcept { return index * 2 + 1; }
    };

    // Generator first, then arch
    const auto idx_re =
        xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, even_idx, dbl_batch::arch_type>().as_batch();
    const auto idx_im =
        xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, odd_idx, dbl_batch::arch_type>().as_batch();

    bench.run("xsimd complex mul (gather/scatter) N=" + std::to_string(N), [&] {
#pragma GCC ivdep
        for (std::size_t i = 0; i < 2 * N; i += dbl_batch::size * 2) {
            auto a_re = dbl_batch::gather(reinterpret_cast<const double*>(a.data()) + i, idx_re);
            auto a_im = dbl_batch::gather(reinterpret_cast<const double*>(a.data()) + i, idx_im);
            auto b_re = dbl_batch::gather(reinterpret_cast<const double*>(b.data()) + i, idx_re);
            auto b_im = dbl_batch::gather(reinterpret_cast<const double*>(b.data()) + i, idx_im);

            auto real = xsimd::fnma(a_im, b_im, a_re * b_re);
            auto imag = xsimd::fma(a_re, b_im, a_im * b_re);

            real.scatter(reinterpret_cast<double*>(out.data()) + i, idx_re);
            imag.scatter(reinterpret_cast<double*>(out.data()) + i, idx_im);
        }
        ankerl::nanobench::doNotOptimizeAway(out);
    });
}

void benchmark_fma_add_sub(std::size_t N, vector_t& a, vector_t& b, vector_t& out, ankerl::nanobench::Bench& bench) {
    using pack = xsimd::batch<double>;  // 4 doubles on AVX2
    using arch = pack::arch_type;

    // --- compile-time helpers -------------------------------------------------
    struct swap_pair {
        static constexpr unsigned get(unsigned i, unsigned) noexcept { return i ^ 1u; }
    };
    struct dup_real {
        static constexpr unsigned get(unsigned i, unsigned) noexcept { return i & ~1u; }
    };
    struct dup_imag {
        static constexpr unsigned get(unsigned i, unsigned) noexcept { return i | 1u; }
    };
    struct odd_lane {
        static constexpr bool get(unsigned i, unsigned) noexcept { return i & 1u; }
    };

    constexpr auto swap_idx = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, swap_pair, arch>();
    constexpr auto real_idx = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, dup_real, arch>();
    constexpr auto imag_idx = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<double>, dup_imag, arch>();

    bench.run("xsimd fmas N=" + std::to_string(N), [&] {
        auto ap = reinterpret_cast<const double*>(a.data());
        auto bp = reinterpret_cast<const double*>(b.data());
        auto op = reinterpret_cast<double*>(out.data());
        constexpr std::size_t S = pack::size;  // 4
        const std::size_t D = N * 2;           // total doubles
        std::size_t i = 0;
#pragma GCC ivdep
        for (; i + 3 < D; i += 4) {  // 4 doubles = 2 complex<double>
            // 1. load [re0,im0,re1,im1]
            const auto va = pack::load_aligned(ap + i);
            const auto vb = pack::load_aligned(bp + i);

            // 2. duplicate real & imag parts of b
            const auto vb_im = xsimd::swizzle(vb, imag_idx);  // [bi0,bi0,bi1,bi1]

            // 3. cross = (ai * bi, ar * bi, …)   using one swizzle on a
            const auto va_sw = xsimd::swizzle(va, swap_idx);
            const auto cross = va_sw * vb_im;

            const auto vb_re = xsimd::swizzle(vb, real_idx);  // [br0,br0,br1,br1]

            const auto result = xsimd::fmas(vb_re, va, cross);

            result.store_aligned(op + i);
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
        for (; i + 3 < D; i += 4) {  // 4 doubles = 2 complex<double>

            const auto va = _mm256_load_pd(ap + i);  // [ar0 ai0 ar1 ai1]
            const auto vb = _mm256_load_pd(bp + i);  // [br0 bi0 br1 bi1]

            const auto vb_re = _mm256_permute_pd(vb, 0x0);  // duplicate re parts
            const auto vb_im = _mm256_permute_pd(vb, 0xF);  // duplicate im parts
            const auto va_sw = _mm256_permute_pd(va, 0x5);  // swap   ar↔ai

            const auto cross = _mm256_mul_pd(va_sw, vb_im);  // ai*bi / ar*bi
            const auto result = _mm256_fmaddsub_pd(vb_re, va, cross);
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