#include "kernelgen/dispatch.hpp"

#include <iostream>
#include <tuple>
#include <utility>

namespace {

struct ExampleKernel {
  template <int Tile, int Vec>
  int operator()(int base) const {
    return (Tile + 1) * 100 + (Vec + 1) * base;
  }
};

struct ExamplePrinter {
  template <int Tile, int Vec>
  void operator()(int value) const {
    std::cout << "Invoked kernel<" << Tile << ", " << Vec << "> with value " << value
              << '\n';
  }
};

} // namespace

int main() {
  auto params = std::make_tuple(
      kernelgen::DispatchParam<kernelgen::make_range<0, 2>>{1},
      kernelgen::DispatchParam<std::integer_sequence<int, 2, 4, 8>>{4});

  ExampleKernel kernel{};
  int result = kernelgen::dispatch(kernel, params, 5);
  std::cout << "Selected kernel returned: " << result << '\n';

  auto printer_params = std::make_tuple(
      kernelgen::DispatchParam<kernelgen::make_range<0, 1>>{0},
      kernelgen::DispatchParam<std::integer_sequence<int, 1, 3>>{3});

  ExamplePrinter printer{};
  kernelgen::dispatch(printer, printer_params, 10);

  auto missing_params = std::make_tuple(
      kernelgen::DispatchParam<kernelgen::make_range<0, 1>>{1},
      kernelgen::DispatchParam<std::integer_sequence<int, 2, 4>>{3});

  int default_result = kernelgen::dispatch(kernel, missing_params, 7);
  std::cout << "Dispatch without match defaults to: " << default_result << '\n';
}

