#include <nanobench.h>

#include <chrono>
#include <xsimd/xsimd.hpp>
constexpr std::size_t iters = 1 << 16;

using arch = xsimd::sse2;

/* --------------------------------------------------------------------- */
/* helper: measure one swizzle pattern                                   */
/* --------------------------------------------------------------------- */
template <class BATCH, class MASK>
void bench_swizzle(ankerl::nanobench::Bench& bench, std::string const& name, MASK const& mask) {
    using batch_t = BATCH;


    bench.run(name, [&] {
        batch_t acc = batch_t(0);
        for (std::size_t i = 0; i < iters; ++i) {
            batch_t v(static_cast<typename batch_t::value_type>(i));
            acc += xsimd::kernel::swizzle(v, mask, arch{});
        }
        ankerl::nanobench::doNotOptimizeAway(acc);
    });
}

int main() {
    ankerl::nanobench::Bench bench;
    using namespace std::chrono_literals;
    bench.title("xsimd::kernel::swizzle  (SSE2)").unit("calls").minEpochTime(100ms).batch(iters);

    /* ---------- float (4 × 32-bit lanes) ----------------------------- */
    using bf = xsimd::batch<float, arch>;
    bench_swizzle<bf>(bench, "float identity", xsimd::batch_constant<uint32_t, arch, 0, 1, 2, 3>{});
    bench_swizzle<bf>(bench, "float dup-lo", xsimd::batch_constant<uint32_t, arch, 0, 1, 0, 1>{});
    bench_swizzle<bf>(bench, "float dup-hi", xsimd::batch_constant<uint32_t, arch, 2, 3, 2, 3>{});
    bench_swizzle<bf>(bench, "float broadcast", xsimd::batch_constant<uint32_t, arch, 3, 3, 3, 3>{});
    bench_swizzle<bf>(bench, "float scramble 0-3-1-2", xsimd::batch_constant<uint32_t, arch, 0, 3, 1, 2>{});

    /* ---------- int32_t (4 × 32-bit lanes) --------------------------- */
    using bi32 = xsimd::batch<int32_t, arch>;
    bench_swizzle<bi32>(bench, "int32 dup-lo", xsimd::batch_constant<uint32_t, arch, 0, 1, 0, 1>{});
    bench_swizzle<bi32>(bench, "int32 scramble 0-3-1-2", xsimd::batch_constant<uint32_t, arch, 0, 3, 1, 2>{});

    /* ---------- double (2 × 64-bit lanes) ---------------------------- */
    using bd = xsimd::batch<double, arch>;
    bench_swizzle<bd>(bench, "double dup-hi", xsimd::batch_constant<uint64_t, arch, 1, 1>{});
    bench_swizzle<bd>(bench, "double swap", xsimd::batch_constant<uint64_t, arch, 1, 0>{});
}
