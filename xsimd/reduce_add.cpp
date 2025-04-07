#include <xsimd/xsimd.hpp>
#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>
#undef ANKERL_NANOBENCH_IMPLEMENT

#include <array>
#include <random>

struct select_even {
    static constexpr unsigned get(unsigned index, unsigned /*size*/) { return index * 2; }
};
struct select_odd {
    static constexpr unsigned get(unsigned index, unsigned /*size*/) { return index * 2 + 1; }
};

struct deinterleave {
    static constexpr unsigned get(unsigned index, unsigned size) { return (index % 2) * (size / 2) + (index / 2); }
};

template <class arch_t, typename T>
constexpr auto select_even_mask = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>, arch_t, select_even>();
template <class arch_t, typename T>
constexpr auto select_odd_mask = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>, arch_t, select_odd>();
template <class arch_t, typename T>
constexpr auto deinterleave_mask = xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>, arch_t, deinterleave>();

namespace xs = xsimd;

template <typename T>
auto xsimd_to_array(const T& vec) noexcept {
    constexpr auto alignment = T::arch_type::alignment();
    alignas(alignment) std::array<typename T::value_type, T::size> array{};
    vec.store_aligned(array.data());
    return array;
}

template <int... i>
constexpr auto hsum(const auto& v, auto& t, std::integer_sequence<int, i...>) noexcept {
    t[0] = (v.get(i * 2) + ...);
    t[1] = (v.get(i * 2 + 1) + ...);
}

int main(const int argc, const char* argv[]) {
    using batch_type = xs::batch<double>;
    constexpr auto size = batch_type::size;
    std::array<double, size * 2> data{};
    std::default_random_engine rng{};
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& d : data) {
        d = dist(rng);
    }
    auto a = batch_type::load_unaligned(data.data());
    auto b = batch_type::load_unaligned(data.data() + size);
    std::array<double, 2> out{0};
    ankerl::nanobench::Bench().run("add+store", [&] {
        const auto res = a + b;
        for (size_t i = 0; i < size; i += 2) {
            out[0] = res.get(i);
            out[1] = res.get(i + 1);
        }
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    ankerl::nanobench::Bench().run("hsum", [&] constexpr noexcept {
        hsum((a + b), out, std::make_integer_sequence<int, size / 2>{});
        ankerl::nanobench::doNotOptimizeAway(out);
    });

    ankerl::nanobench::Bench().run("reduce_add", [&] constexpr noexcept {
        const auto res_real = xsimd::shuffle(a, b, select_even_mask<typename batch_type::arch_type, double>);
        const auto res_imag = xsimd::shuffle(a, b, select_odd_mask<typename batch_type::arch_type, double>);
        out[0] = xsimd::reduce_add(res_real);
        out[1] = xsimd::reduce_add(res_imag);
        ankerl::nanobench::doNotOptimizeAway(out);
    });

    ankerl::nanobench::Bench().run("union pun", [&] constexpr noexcept {
        const auto res = a + b;
        const auto low_high = xsimd::swizzle(res, deinterleave_mask<typename batch_type::arch_type, double>);
        using half = xsimd::make_sized_batch_t<double, size / 2>;
        static_assert(!std::is_void_v<half>);
        union {
            struct {
                half low;
                half high;
            } vec;
            batch_type all;
        } pun = {.all = low_high};
        out[0] = xsimd::reduce_add(pun.vec.low);
        out[1] = xsimd::reduce_add(pun.vec.high);
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    if constexpr (size >= 8) {
        ankerl::nanobench::Bench().run("double union pun", [&] constexpr noexcept {
            const auto res = a + b;
            using half = xsimd::make_sized_batch_t<double, size / 2>;
            static_assert(!std::is_void_v<half>);
            union {
                struct {
                    half low;
                    half high;
                } vec;
                batch_type all;
            } pun = {.all = res};
            const auto res2 = pun.vec.low + pun.vec.high;
            const auto low_high = xsimd::swizzle(res2, deinterleave_mask<typename half::arch_type, double>);
            using quarter = xsimd::make_sized_batch_t<double, size / 4>;
            static_assert(!std::is_void_v<quarter>);
            union {
                struct {
                    quarter low;
                    quarter high;
                } vec;
                half all;
            } pun2 = {.all = low_high};
            out[0] = xsimd::reduce_add(pun2.vec.low);
            out[1] = xsimd::reduce_add(pun2.vec.high);
            ankerl::nanobench::doNotOptimizeAway(out);
        });
    }
}