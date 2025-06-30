#include <xsimd/xsimd.hpp>
#include <nanobench.h>

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
    const auto vec = xsimd_to_array(v);
    t[0] = (vec[i * 2] + ...);
    t[1] = (vec[i * 2 + 1] + ...);
}

template <int... i>
constexpr auto hsum(const auto v, std::integer_sequence<int, i...>) noexcept {
    return (v.data[i] + ...);
}
constexpr auto hsum(const auto v) noexcept { return hsum(v, std::make_integer_sequence<int, decltype(v)::size>{}); }

template <int... i>
constexpr auto shsum(const auto& v, auto& t, std::integer_sequence<int, i...>) noexcept {
    const auto low_high = xsimd::swizzle(v, deinterleave_mask<typename std::decay_t<decltype(v)>::arch_type, double>);
    t[0] = (low_high.data[i] + ...);
    t[1] = (low_high.data[i + sizeof...(i)] + ...);
}

template <typename Batch>
constexpr auto complex_hadd(const Batch& res) {
    constexpr std::size_t size = Batch::size;
    if constexpr (size == 2) {
        return res;
    } else {
        using half_t = xsimd::make_sized_batch_t<typename Batch::value_type, size / 2>;
        static_assert(!std::is_void_v<half_t>, "xsimd does not support this batch size.");
        alignas(Batch::arch_type::alignment()) std::array<half_t, 2> out{};
        res.store_aligned(out.data());
        return complex_hadd(out[0]+out[1]);
    }
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
    alignas(batch_type::arch_type::alignment()) std::array<double, 2> out{};
    ankerl::nanobench::Bench().run("add+store", [&] {
        const auto res = a + b;
        for (size_t i = 0; i < size; i += 2) {
            out[0] += res.data[i];
            out[1] += res.data[i+1];
        }
    });
    const auto res = a + b;
    for (size_t i = 0; i < size; i += 2) {
        out[0] += res.get(i);
        out[1] += res.get(i + 1);
    }
    if (argc > 1) {
        std::cout << "Result: " << out[0] << ", " << out[1] << "\n";
    }
    ankerl::nanobench::Bench().run("hsum", [&] constexpr noexcept {
        hsum((a + b), out, std::make_integer_sequence<int, size / 2>{});
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    if (argc > 1) {
        std::cout << "Result: " << out[0] << ", " << out[1] << "\n";
    }
    ankerl::nanobench::Bench().run("shsum", [&] constexpr noexcept {
        shsum((a + b), out, std::make_integer_sequence<int, size / 2>{});
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    if (argc > 1) {
        std::cout << "Result: " << out[0] << ", " << out[1] << "\n";
    }
    ankerl::nanobench::Bench().run("reduce_add", [&] constexpr noexcept {
        const auto res_real = xsimd::shuffle(a, b, select_even_mask<typename batch_type::arch_type, double>);
        const auto res_imag = xsimd::shuffle(a, b, select_odd_mask<typename batch_type::arch_type, double>);
        out[0] = xsimd::reduce_add(res_real);
        out[1] = xsimd::reduce_add(res_imag);
        ankerl::nanobench::doNotOptimizeAway(out);
    });
    if (argc > 1) {
        std::cout << "Result: " << out[0] << ", " << out[1] << "\n";
    }
    ankerl::nanobench::Bench().run("complex hsum", [&] constexpr noexcept {
        const auto vec = complex_hadd(a + b);
        vec.store_aligned(out.data());
    });
    if (argc > 1) {
        std::cout << "Result: " << out[0] << ", " << out[1] << "\n";
    }

}