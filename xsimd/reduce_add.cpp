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
    static constexpr unsigned get(unsigned index, unsigned /*size*/) {
        return index * 2 + 1;
    }
};

template<class arch_t, typename T>
constexpr auto select_even_mask =
    xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>, arch_t, select_even>();
template<class arch_t, typename T>
constexpr auto select_odd_mask =
    xsimd::make_batch_constant<xsimd::as_unsigned_integer_t<T>, arch_t, select_odd>();

namespace xs = xsimd;

template<typename T>  auto xsimd_to_array(const T &vec) noexcept {
    constexpr auto alignment = T::arch_type::alignment();
    alignas(alignment) std::array<typename T::value_type, T::size> array{};
    vec.store_aligned(array.data());
    return array;
}

int main(const int argc, const char* argv[])
{
    using batch_type = xs::batch<double>;
    constexpr auto size = xs::batch<double>::size;
    std::array<double, size*2> data{};
    std::default_random_engine rng{};
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (auto& d : data) {
        d = dist(rng);
    }
    auto a = xs::load_unaligned(data.data());
    auto b = xs::load_unaligned(data.data() + size);
    std::array<double, 2> out{0};
    ankerl::nanobench::Bench().run("add+store", [&] {
        const auto res = a + b;
        const auto res_array = xsimd_to_array(res);
        for (size_t i = 0; i < size; i+=2) {
            out[0] = res_array[i];
            out[1] = res_array[i+1];
        }
        ankerl::nanobench::doNotOptimizeAway(out);
    });

    ankerl::nanobench::Bench().run("reduce_add", [&] {
        const auto res_real = xsimd::shuffle(a, b, select_even_mask<typename xs::batch<double>::arch_type, double>);
        const auto res_imag = xsimd::shuffle(a, b, select_odd_mask<typename xs::batch<double>::arch_type, double>);
        out[0]              = xsimd::reduce_add(res_real);
        out[1]              = xsimd::reduce_add(res_imag);
        ankerl::nanobench::doNotOptimizeAway(out);
    });

}