// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "highway.cpp"  // this file
#include <hwy/foreach_target.h>  // must come before highway.h
#include <hwy/highway.h>

#include <benchmark/benchmark.h>

#include <array>
#include <random>
#include <chrono>
#include <iostream>

HWY_BEFORE_NAMESPACE();  // required if not using HWY_ATTR
namespace project {
namespace HWY_NAMESPACE {  // required: unique per target

// Can skip hn:: prefixes if already inside hwy::HWY_NAMESPACE.
namespace hn = hwy::HWY_NAMESPACE;

using T = float;
void MulAddLoop(const T* HWY_RESTRICT mul_array,
                const T* HWY_RESTRICT add_array,
                const size_t size, T* HWY_RESTRICT x_array) {
  const hn::ScalableTag<T> d;
  // std::cerr << __PRETTY_FUNCTION__ << std::endl;
// #pragma clang loop vectorize(enable) interleave(enable)
    for (size_t i = 0; i < size; i += hn::Lanes(d)) {
      const auto mul = hn::Load(d, mul_array + i);
      const auto add = hn::Load(d, add_array + i);
      auto x = hn::Load(d, x_array + i);
      x = hn::MulAdd(mul, x, add);
      hn::Store(x, d, x_array + i);
    }
}

}  // namespace HWY_NAMESPACE

}  // namespace project
HWY_AFTER_NAMESPACE();

// The table of pointers to the various implementations in HWY_NAMESPACE must
// be compiled only once (foreach_target #includes this file multiple times).
// HWY_ONCE is true for only one of these 'compilation passes'.
#if HWY_ONCE

namespace project {


// This macro declares a static array used for dynamic dispatch.
HWY_EXPORT(MulAddLoop);

void CallMulAddLoop(const float*  mul_array,
                const float*  add_array,
                const size_t size, float*  x_array) {
  // This must reside outside of HWY_NAMESPACE because it references (calls the
  // appropriate one from) the per-target implementations there.
  // For static dispatch, use HWY_STATIC_DISPATCH.
  return HWY_DYNAMIC_DISPATCH(MulAddLoop)(mul_array, add_array, size, x_array);
}

void MulAddVector(const float*  mul_array,
                  const float*  add_array,
                  const size_t size, float*  x_array) {
  // std::cerr << __PRETTY_FUNCTION__ << std::endl;
#pragma clang loop vectorize(enable) interleave(enable)
    for (size_t i = 0; i < size; ++i) {
      x_array[i] = mul_array[i] * x_array[i] + add_array[i];
    }
}


void MulAddScalar(const float*  mul_array,
                  const float*  add_array,
                  const size_t size, float*  x_array) {
  // std::cerr << __PRETTY_FUNCTION__ << std::endl;
#pragma clang loop vectorize(disable) interleave(disable)
    for (size_t i = 0; i < size; ++i) {
      x_array[i] = mul_array[i] * x_array[i] + add_array[i];
    }
}

}  // namespace project

namespace {
  static constexpr auto size = 1<<20;
  alignas(32) std::array<float, size> mul_array;
  alignas(32) std::array<float, size> add_array;
  alignas(32) std::array<float, size> x_array;
}

void HwyVector(benchmark::State& state) {
  for (auto _ : state){
    project::CallMulAddLoop(mul_array.data(), add_array.data(), size, x_array.data());
  }
}

void Scalar(benchmark::State& state) {
  for (auto _ : state){
    project::MulAddScalar(mul_array.data(), add_array.data(), size, x_array.data());
  }
}

void Vector(benchmark::State& state) {
  state.ResumeTiming();
  for (auto _ : state){
    project::MulAddVector(mul_array.data(), add_array.data(), size, x_array.data());
  }
}

BENCHMARK(Scalar)->Iterations(100);
BENCHMARK(Vector)->Iterations(100);
BENCHMARK(HwyVector)->Iterations(100);

int main(int argc, char** argv) {
  std::random_device rd;
  std::uniform_real_distribution<float> dist{0.0, 1.0};
  for (size_t j = 0; j < 1; j++){
    for (size_t i = 0; i < size; ++i) {
      mul_array[i] = dist(rd);
      add_array[i] = dist(rd);
      x_array[i] = dist(rd);
    }                                     
  }
  
  char arg0_default[] = "benchmark";                                  
  char* args_default = arg0_default;                                  
  if (!argv) {                                                        
    argc = 1;                                                         
    argv = &args_default;                                             
  }                                                                   
  benchmark::Initialize(&argc, argv);                               
  if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; 
  benchmark::RunSpecifiedBenchmarks();                              
  benchmark::Shutdown();                                            
  return 0;                                                           
}   

#endif  // HWY_ONCE
