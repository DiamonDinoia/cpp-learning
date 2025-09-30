// bench.cpp
// Build: g++-13 -O3 -march=native -std=c++17 -DNDEBUG -I/path/to/nanobench bench.cpp -o bench
#include <nanobench.h>

#include <iostream>

#include "dispatch.h"

int main() {
    using namespace std::chrono_literals;
    using S1 = make_range<1, 6>;                               // contiguous
    using S2 = std::integer_sequence<int, 3, 4, 7, 8, 9, 10>;  // non-contiguous
    using S3 = make_range<1, 5>;
    using S4 = std::integer_sequence<int, -5, -3, 0, 2, 11>;  // non-contiguous
    using S5 = make_range<2, 7>;

    auto params = std::make_tuple(DispatchParam<S1>{1}, DispatchParam<S2>{7}, DispatchParam<S3>{3},
                                  DispatchParam<S4>{0}, DispatchParam<S5>{4});

    FuncKernel k;

    // sanity
    const int probe = 123;
    const int r_lin = linear_dispatch(k, params, probe);
    const int r_vis = visit_dispatch(k, params, probe);
    const int r_tab = table_dispatch(k, params, probe);
    if (!(r_lin == r_vis && r_vis == r_tab)) {
        std::cerr << "Mismatch\n";
        return 1;
    }

    ankerl::nanobench::Bench bench;
    bench.title("N-ary dispatch on mixed contiguous/non-contiguous sets")
        .unit("call")
        .warmup(50)
        .minEpochTime(200ms)
        .minEpochIterations(50)
        .relative(true);

    int x = 0;
    bench.run("linear-scan", [&] {
        auto v = linear_dispatch(k, params, ++x);
        ankerl::nanobench::doNotOptimizeAway(v);
    });
    bench.run("variant + visit", [&] {
        auto v = visit_dispatch(k, params, ++x);
        ankerl::nanobench::doNotOptimizeAway(v);
    });
    bench.run("jump-table", [&] {
        auto v = table_dispatch(k, params, ++x);
        ankerl::nanobench::doNotOptimizeAway(v);
    });

    return 0;
}
