#include "kernelgen/unroll.hpp"

#include <cstddef>
#include <iostream>

int main() {
  std::cout << "Static loop range [0, 6) with stride 2:" << '\n';
  kernelgen::static_loop<0, 6, 2>([](auto index) {
    constexpr std::size_t value = static_cast<std::size_t>(index);
    std::cout << "  iteration " << value << '\n';
  });

  std::cout << "Static loop range [0, 4) with default stride:" << '\n';
  kernelgen::static_loop<4>([](auto index) {
    constexpr std::size_t value = static_cast<std::size_t>(index);
    std::cout << "  iteration " << value << '\n';
  });

  std::cout << "Static loop range [1, 5) with default stride:" << '\n';
  kernelgen::static_loop<1, 5>([](auto index) {
    constexpr std::size_t value = static_cast<std::size_t>(index);
    std::cout << "  iteration " << value << '\n';
  });

  constexpr std::size_t unroll_factor = 3;
  std::cout << "Unroll loop over [0, 10) with factor " << unroll_factor << ':' << '\n';
  constexpr std::size_t static_start  = 0;
  constexpr std::size_t static_stop   = 10;
  constexpr std::size_t static_stride = 1;
  constexpr std::size_t static_total =
      kernelgen::compute_range_count<static_start, static_stop, static_stride>;
  constexpr std::size_t static_chunk_count     = static_total / unroll_factor;
  constexpr std::size_t static_chunk_coverage = static_chunk_count * unroll_factor;
  kernelgen::unroll_loop<static_start, static_stop, static_stride, unroll_factor>(
      [&](auto index) {
        constexpr std::size_t value = static_cast<std::size_t>(index);
        constexpr bool is_tail      = value >= static_chunk_coverage;
        constexpr std::size_t chunk = value / unroll_factor;
        std::cout << "  iteration " << value;
        if constexpr (is_tail) {
          std::cout << " (tail)";
        } else {
          std::cout << " (chunk " << chunk << ')';
        }
        std::cout << '\n';
      });

  constexpr std::size_t runtime_start  = 2;
  constexpr std::size_t runtime_stop   = 17;
  constexpr std::size_t runtime_stride = 4;
  std::cout << "Runtime unroll loop over [" << runtime_start << ", " << runtime_stop
            << ") stride " << runtime_stride << " with factor " << unroll_factor
            << ':' << '\n';
  constexpr std::size_t runtime_total = kernelgen::compute_range_count_runtime(
      runtime_start, runtime_stop, runtime_stride);
  constexpr std::size_t runtime_chunk_count = runtime_total / unroll_factor;
  constexpr std::size_t runtime_chunk_coverage = runtime_chunk_count * unroll_factor;
  std::size_t runtime_invocation = 0;
  kernelgen::unroll_loop<unroll_factor>(runtime_start,
                                        runtime_stop,
                                        runtime_stride,
                                        [&](std::size_t index) {
                                          const bool is_tail =
                                              runtime_invocation >= runtime_chunk_coverage;
                                          const std::size_t chunk = runtime_invocation / unroll_factor;
                                          std::cout << "  runtime iteration " << runtime_invocation
                                                    << " index " << index;
                                          if (is_tail) {
                                            std::cout << " (tail)";
                                          } else {
                                            std::cout << " (chunk " << chunk << ')';
                                          }
                                          std::cout << '\n';
                                          ++runtime_invocation;
                                        });

  auto run_runtime_implicit = [&](std::size_t start_value,
                                  std::size_t stop_value,
                                  std::size_t stride_value) {
    const std::size_t total_iterations = kernelgen::compute_range_count_runtime(
        start_value, stop_value, stride_value);
    const std::size_t chunk_count = total_iterations / unroll_factor;
    const std::size_t chunk_coverage = chunk_count * unroll_factor;
    const std::size_t expected_tail = total_iterations - chunk_coverage;

    std::cout << "Runtime unroll loop over [" << start_value << ", " << stop_value
              << ") stride " << stride_value
              << " with implicit tail (expected tail length " << expected_tail << ")"
              << ':' << '\n';

    std::size_t invocation_index = 0;
    std::size_t tail_hits        = 0;
    kernelgen::unroll_loop<unroll_factor>(
        start_value,
        stop_value,
        stride_value,
        [&](std::size_t index) {
          const bool is_tail = invocation_index >= chunk_coverage;
          std::cout << "  implicit iteration " << invocation_index << " index " << index;
          if (is_tail) {
            std::cout << " (tail)";
            ++tail_hits;
          }
          std::cout << '\n';
          ++invocation_index;
        });

    std::cout << "  observed tail iterations: " << tail_hits << " of " << expected_tail
              << '\n';
  };

  run_runtime_implicit(runtime_start, runtime_stop, runtime_stride);
  run_runtime_implicit(std::size_t{0}, std::size_t{20}, std::size_t{4});
  run_runtime_implicit(std::size_t{0}, std::size_t{2}, std::size_t{1});
}

