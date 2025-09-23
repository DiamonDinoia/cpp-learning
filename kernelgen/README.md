# kernelgen

`kernelgen` provides header-only utilities for building compile-time driven kernels in C++17.

## Features

- **Template dispatch** – map runtime integers to compile-time template parameters using cartesian-product
  enumeration.
- **Static loop helpers** – expand loops at compile time and invoke a functor with `std::integral_constant`
  indices via `static_loop`.
- **Loop unrolling** – duplicate the loop body with a compile-time unroll factor while automatically
  handling remainder iterations.

All APIs live in the `kernelgen` namespace and require only a C++17 toolchain.

## Usage

```cpp
#include <tuple>
#include "kernelgen/dispatch.hpp"

struct Functor {
  template<int I, int J>
  int operator()(int value) const { return value + I * 10 + J; }
};

int main() {
  auto params = std::make_tuple(
      kernelgen::DispatchParam<kernelgen::make_range<0, 2>>{1},
      kernelgen::DispatchParam<std::integer_sequence<int, 0, 4>>{4});

  Functor f{};
  int result = kernelgen::dispatch(f, params, 5);
}
```

For static loops, simply provide the compile-time bounds and a callable:

```cpp
#include "kernelgen/unroll.hpp"

kernelgen::static_loop<0, 8, 2>([](auto index) {
  // index is std::integral_constant<std::size_t, ...>
});
```

To unroll a compile-time loop with a factor, provide a callable and the remainder iterations will be
covered automatically:

```cpp
constexpr std::size_t factor = 4;
kernelgen::unroll_loop<0, 10, 1, factor>([](auto idx) {
  // body invoked once per logical iteration; the remainder runs automatically.
});
```

When the iteration bounds are only known at runtime, use the overload that accepts the range and
stride as values while keeping the unroll factor a compile-time constant:

```cpp
constexpr std::size_t factor = 4;
std::size_t start  = 2;
std::size_t stop   = 17;
std::size_t stride = 3;
kernelgen::unroll_loop<factor>(
    start,
    stop,
    stride,
    [](std::size_t idx) {
      // body executes once per logical iteration; tail iterations reuse the same callable.
    });
```

Refer to the programs in `kernelgen/samples` for runnable demonstrations.
