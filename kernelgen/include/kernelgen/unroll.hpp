#pragma once

#include <cassert>
#include <cstddef>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>

#include "kernelgen/dispatch.hpp"

namespace kernelgen {

template <std::size_t Start, std::size_t Stop, std::size_t Inc>
inline constexpr std::size_t compute_range_count =
    (Start < Stop ? ((Stop - Start + Inc - 1) / Inc) : 0);

constexpr std::size_t compute_range_count_runtime(
    std::size_t start, std::size_t stop, std::size_t inc) {
  return (start < stop && inc > 0) ? ((stop - start + inc - 1) / inc) : 0;
}

namespace detail {
template <std::size_t Start, std::size_t Inc, typename F, std::size_t... Is>
constexpr void static_loop_impl(F &&f, std::index_sequence<Is...>) {
  (f(std::integral_constant<std::size_t, Start + Is * Inc>{}), ...);
}
} // namespace detail

template <std::size_t Start, std::size_t Stop, std::size_t Inc = 1, typename F>
constexpr void static_loop(F &&f) {
  static_assert(Inc > 0, "kernelgen::static_loop requires Inc > 0");
  constexpr std::size_t count = compute_range_count<Start, Stop, Inc>;
  detail::static_loop_impl<Start, Inc>(std::forward<F>(f),
                                       std::make_index_sequence<count>{});
}

template <std::size_t Stop, typename F> constexpr void static_loop(F &&f) {
  static_loop<0, Stop, 1>(std::forward<F>(f));
}

namespace detail {
template <std::size_t Start, std::size_t Inc, std::size_t Count, typename F>
constexpr void unroll_loop_apply(F &&f) {
  static_loop<Start, Start + Count * Inc, Inc>(std::forward<F>(f));
}
} // namespace detail

namespace detail {
template <std::size_t Start,
          std::size_t Stop,
          std::size_t Inc,
          std::size_t Factor,
          typename Body>
constexpr void compile_time_unroll(Body &&body) {
  static_assert(Factor > 0, "kernelgen::unroll_loop requires Factor > 0");
  static_assert(Inc > 0, "kernelgen::unroll_loop requires Inc > 0");

  constexpr std::size_t total_iterations = compute_range_count<Start, Stop, Inc>;
  constexpr std::size_t chunk_iterations = total_iterations / Factor;
  constexpr std::size_t tail_iterations  = total_iterations % Factor;

  auto &&body_fn = std::forward<Body>(body);

  static_loop<0, chunk_iterations>([&](auto chunk) {
    constexpr std::size_t chunk_index = static_cast<std::size_t>(chunk);
    constexpr std::size_t chunk_start = Start + chunk_index * Factor * Inc;
    detail::unroll_loop_apply<chunk_start, Inc, Factor>(body_fn);
  });

  if constexpr (tail_iterations > 0) {
    constexpr std::size_t tail_start = Start + chunk_iterations * Factor * Inc;
    detail::unroll_loop_apply<tail_start, Inc, tail_iterations>(body_fn);
  }
}
} // namespace detail

template <std::size_t Start,
          std::size_t Stop,
          std::size_t Inc,
          std::size_t Factor,
          typename Body>
constexpr void unroll_loop(Body &&body) {
  detail::compile_time_unroll<Start, Stop, Inc, Factor>(std::forward<Body>(body));
}

namespace detail {
template <std::size_t Factor, typename Body, std::size_t... Offsets>
constexpr void runtime_unroll_apply(Body &&body,
                                    std::size_t base,
                                    std::size_t inc,
                                    std::index_sequence<Offsets...>) {
  ((void)std::forward<Body>(body)(base + Offsets * inc), ...);
}

template <typename Body> struct runtime_tail_dispatcher {
  Body *body;
  std::size_t base;
  std::size_t stride;

  template <int TailCount> void operator()() const {
    constexpr std::size_t count = static_cast<std::size_t>(TailCount);
    detail::runtime_unroll_apply<count>(*body, base, stride,
                                        std::make_index_sequence<count>{});
  }
};
} // namespace detail

template <std::size_t Factor, typename Body>
constexpr void unroll_loop(std::size_t start,
                           std::size_t stop,
                           std::size_t inc,
                           Body &&body) {
  static_assert(Factor > 0, "kernelgen::unroll_loop requires Factor > 0");
  assert(inc > 0 && "kernelgen::unroll_loop requires Inc > 0");

  if (inc == 0) return;

  const std::size_t total_iterations =
      compute_range_count_runtime(start, stop, inc);
  const std::size_t chunk_iterations = total_iterations / Factor;
  const std::size_t tail_iterations  = total_iterations % Factor;

  auto &&body_fn = std::forward<Body>(body);

  std::size_t current = start;
  for (std::size_t chunk = 0; chunk < chunk_iterations; ++chunk) {
    detail::runtime_unroll_apply<Factor>(body_fn, current, inc,
                                         std::make_index_sequence<Factor>{});
    current += Factor * inc;
  }

  if (tail_iterations > 0) {
    const std::size_t tail_base = current;

    if constexpr (Factor > 1) {
      static_assert(Factor <= static_cast<std::size_t>(std::numeric_limits<int>::max()),
                    "kernelgen::unroll_loop tail dispatch requires Factor to fit in int");

      using tail_range = make_range<1, static_cast<int>(Factor) - 1>;
      using body_type  = std::remove_reference_t<decltype(body_fn)>;
      detail::runtime_tail_dispatcher<body_type> tail_dispatcher{
          std::addressof(body_fn), tail_base, inc};
      dispatch(tail_dispatcher,
               std::make_tuple(DispatchParam<tail_range>{
                   static_cast<int>(tail_iterations)}));
    }
  }
}

template <std::size_t Factor, typename Body>
constexpr void unroll_loop(std::size_t start, std::size_t stop, Body &&body) {
  unroll_loop<Factor>(start, stop, std::size_t{1}, std::forward<Body>(body));
}

template <std::size_t Factor, typename Body>
constexpr void unroll_loop(std::size_t stop, Body &&body) {
  unroll_loop<Factor>(std::size_t{0}, stop, std::forward<Body>(body));
}

template <std::size_t Start, std::size_t Stop, std::size_t Factor, typename Body>
constexpr void unroll_loop(Body &&body) {
  detail::compile_time_unroll<Start, Stop, 1, Factor>(std::forward<Body>(body));
}

template <std::size_t Stop, std::size_t Factor, typename Body>
constexpr void unroll_loop(Body &&body) {
  detail::compile_time_unroll<0, Stop, 1, Factor>(std::forward<Body>(body));
}

} // namespace kernelgen

