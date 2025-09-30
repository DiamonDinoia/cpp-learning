// tests.cpp
// Build: g++-13 -O3 -std=c++17 -DNDEBUG tests.cpp -o tests
#include "dispatch.h"

#include <cassert>
#include <iostream>

// A void-return kernel to exercise the void codepaths
struct VoidKernel {
    template <int... P>
    void operator()(int& acc) const noexcept {
        acc += (0 + ... + P);
    }
};

template <typename Seq>
constexpr auto values_of() {
    return seq_info<Seq>::values();
}

// Test 1: Exhaustive agreement on all combinations (5-ary), multiple x
void test_exhaustive_agreement() {
    using S1 = make_range<1, 2>;
    using S2 = std::integer_sequence<int, 3, 4, 7>;
    using S3 = make_range<1, 4>;
    using S4 = std::integer_sequence<int, -5, -3, 0, 2>;
    using S5 = make_range<2, 5>;

    constexpr auto v1 = values_of<S1>();
    constexpr auto v2 = values_of<S2>();
    constexpr auto v3 = values_of<S3>();
    constexpr auto v4 = values_of<S4>();
    constexpr auto v5 = values_of<S5>();

    FuncKernel k;

    for (int a : v1)
        for (int b : v2)
            for (int c : v3)
                for (int d : v4)
                    for (int e : v5) {
                        auto params = std::make_tuple(DispatchParam<S1>{a}, DispatchParam<S2>{b}, DispatchParam<S3>{c},
                                                      DispatchParam<S4>{d}, DispatchParam<S5>{e});
                        for (int x : {a, b, c, d, e}) {
                            const int r_lin = linear_dispatch(k, params, x);
                            const int r_vis = visit_dispatch(k, params, x);
                            const int r_tab = table_dispatch(k, params, x);
                            if (r_lin != r_vis || r_vis != r_tab || r_lin != (a + b + c + d + e + x)) std::cout << "mismatch\n";
                            assert(r_lin == r_vis);
                            assert(r_vis == r_tab);
                            assert(r_lin == (a + b + c + d + e + x));
                        }
                    }
}

// Test 2: Invalid inputs return default-initialized result (int -> 0)
void test_invalid_values_return_default() {
    using S1 = make_range<1, 6>;
    using S2 = std::integer_sequence<int, 3, 4, 7, 8, 9, 10>;
    using S3 = make_range<1, 5>;
    using S4 = std::integer_sequence<int, -5, -3, 0, 2, 11>;
    using S5 = make_range<2, 7>;

    auto params = std::make_tuple(DispatchParam<S1>{1}, DispatchParam<S2>{5}, DispatchParam<S3>{3},
                                  DispatchParam<S4>{0}, DispatchParam<S5>{4});
    FuncKernel k;
    for (int x : {-10, 0, 42}) {
        const int r_lin = linear_dispatch(k, params, x);
        const int r_vis = visit_dispatch(k, params, x);
        const int r_tab = table_dispatch(k, params, x);
        assert(r_lin == 0);
        assert(r_vis == 0);
        assert(r_tab == 0);
    }
}

// Test 3: Void-return path has identical side effects across dispatchers
void test_void_return_path() {
    using S1 = make_range<1, 3>;
    using S2 = std::integer_sequence<int, 7, 8>;
    using S3 = std::integer_sequence<int, -2, 5>;

    constexpr auto v1 = values_of<S1>();
    constexpr auto v2 = values_of<S2>();
    constexpr auto v3 = values_of<S3>();

    VoidKernel vk;
    for (int a : v1)
        for (int b : v2)
            for (int c : v3) {
                auto params = std::make_tuple(DispatchParam<S1>{a}, DispatchParam<S2>{b}, DispatchParam<S3>{c});
                int acc1 = 0, acc2 = 0, acc3 = 0;
                linear_dispatch(vk, params, acc1);
                visit_dispatch(vk, params, acc2);
                table_dispatch(vk, params, acc3);
                const int expected = a + b + c;
                assert(acc1 == expected);
                assert(acc2 == expected);
                assert(acc3 == expected);
            }

    // Invalid value -> no side effect
    auto bad = std::make_tuple(DispatchParam<S1>{999}, DispatchParam<S2>{7}, DispatchParam<S3>{-2});
    int acc = 123;
    linear_dispatch(vk, bad, acc);
    assert(acc == 123);
    visit_dispatch(vk, bad, acc);
    assert(acc == 123);
    table_dispatch(vk, bad, acc);
    assert(acc == 123);
}

// Test 4: Large span (>64) exercises slow path of map_to_index
void test_large_span_mapping() {
    using Sbig = std::integer_sequence<int, 0, 65, 130>;  // span = 131 > 64
    static_assert(seq_info<Sbig>::span() > 64, "expected large span");
    constexpr auto vb = seq_info<Sbig>::values();

    FuncKernel k;
    for (int v : vb) {
        auto p = std::make_tuple(DispatchParam<Sbig>{v});
        const int x = 11;
        const int r1 = linear_dispatch(k, p, x);
        const int r2 = visit_dispatch(k, p, x);
        const int r3 = table_dispatch(k, p, x);
        assert(r1 == r2 && r2 == r3);
        assert(r1 == v + x);
    }
    auto bad = std::make_tuple(DispatchParam<Sbig>{1});
    const int x = 5;
    const int r1 = linear_dispatch(k, bad, x);
    const int r2 = visit_dispatch(k, bad, x);
    const int r3 = table_dispatch(k, bad, x);
    assert(r1 == 0 && r2 == 0 && r3 == 0);
}

int main() {
    test_exhaustive_agreement();
    test_invalid_values_return_default();
    test_void_return_path();
    test_large_span_mapping();
    std::cout << "All tests passed.\n";
    return 0;
}
