// Generates code for every target that this compiler can support.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "highway.cpp"  // this file
#include <benchmark/benchmark.h>
#include <hwy/foreach_target.h>  // must come before highway.h
#include <hwy/highway.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <valarray>

HWY_BEFORE_NAMESPACE();  // required if not using HWY_ATTR
namespace project {
namespace HWY_NAMESPACE {  // required: unique per target

// Can skip hn:: prefixes if already inside hwy::HWY_NAMESPACE.
namespace hn = hwy::HWY_NAMESPACE;

void MulAddLoop(const float* HWY_RESTRICT mul_array, const float* HWY_RESTRICT add_array, const size_t size,
                float* HWY_RESTRICT x_array) {
    const hn::ScalableTag<float> d;
    // std::cerr << __PRETTY_FUNCTION__ << std::endl;
    for (size_t i = 0; i < size; i += hn::Lanes(d)) {
        const auto mul = hn::Load(d, mul_array + i);
        const auto add = hn::Load(d, add_array + i);
        auto x = hn::Load(d, x_array + i);
        x = hn::MulAdd(mul, x, add);
        hn::Store(x, d, x_array + i);
    }
}

/* compute natural logarithm, maximum error 0.85089 ulps \
 * source
 * https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c
 */
template <typename T>
inline constexpr T fast_logfV(T a) {
    constexpr hn::ScalableTag<float> fl;
    constexpr hn::ScalableTag<std::int32_t> il;
    decltype(hn::Undefined(fl)) i, m, r, s, t;
    decltype(hn::Undefined(il)) e;
    hn::Set(fl, 1.19209290e-7f);
    auto mask = a < hn::Set(fl, 1.175494351e-38f);                     // 0x1.0p-126
    a = hn::IfThenElse(mask, hn::Mul(a, hn::Set(fl, 8388608.0f)), a);  // 0x1.0p+23
    i = hn::IfThenElseZero(mask, hn::Sub(i, hn::Set(fl, 23.0f)));
    e = hn::Sub(hn::BitCast(il, a), hn::BitCast(il, hn::Set(fl, 0.666666667f)));
    e = hn::And(e, hn::Set(il, 0xff800000));
    m = hn::BitCast(fl, hn::Sub(hn::BitCast(il, a), e));
    i = hn::MulAdd(hn::ConvertTo(fl, e), hn::Set(fl, 1.19209290e-7f), i);  // 0x1.0p-23
    /* m in [2/3, 4/3] */
    m = hn::Sub(m, hn::Set(fl, 1.0f));
    s = hn::Mul(m, m);
    /* Compute log1p(m) for m in [-1/3, 1/3] */
    r = hn::Set(fl, -0.130310059f);                   // -0x1.0ae000p-3
    t = hn::Set(fl, 0.140869141f);                    //  0x1.208000p-3
    r = hn::MulSub(r, s, hn::Set(fl, 0.121483512f));  // -0x1.f198b2p-4
    t = hn::MulAdd(t, s, hn::Set(fl, 0.139814854f));  //  0x1.1e5740p-3
    r = hn::MulSub(r, s, hn::Set(fl, 0.166846126f));  // -0x1.55b36cp-3
    t = hn::MulAdd(t, s, hn::Set(fl, 0.200120345f));  //  0x1.99d8b2p-3
    r = hn::MulSub(r, s, hn::Set(fl, 0.249996200f));  // -0x1.fffe02p-3
    r = hn::MulAdd(t, m, r);
    r = hn::MulAdd(r, m, hn::Set(fl, 0.333331972f));  //  0x1.5554fap-2
    r = hn::MulSub(r, m, hn::Set(fl, 0.500000000f));  // -0x1.000000p-1
    r = hn::MulAdd(r, s, m);
    r = hn::MulAdd(i, hn::Set(fl, 0.693147182f), r);  //  0x1.62e430p-1 // log(2)
    mask = hn::Or(a <= hn::Zero(fl), a >= hn::Set(fl, INFINITY));
    r = hn::IfThenElse(mask, hn::Add(a, a), r);  // silence NaNs if necessary
    r = hn::IfThenElse(hn::And(mask, a < hn::Zero(fl)), hn::Set(fl, INFINITY - INFINITY), r);  //  NaN
    r = hn::IfThenElse(hn::And(mask, a == hn::Zero(fl)), hn::Set(fl, -INFINITY), r);           // -Infinty
    return r;
}

//
void LogLoop(const float* HWY_RESTRICT in_array, float* HWY_RESTRICT out_array, const size_t size) {
    const hn::ScalableTag<float> d;
    for (size_t i = 0; i < size; i += hn::Lanes(d)) {
        hn::Store(fast_logfV(hn::Load(d, in_array + i)), d, out_array + i);
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

void CallMulAddLoop(const float* mul_array, const float* add_array, const size_t size, float* x_array) {
    // This must reside outside of HWY_NAMESPACE because it references (calls
    // the appropriate one from) the per-target implementations there. For
    // static dispatch, use HWY_STATIC_DISPATCH.
    return HWY_DYNAMIC_DISPATCH(MulAddLoop)(mul_array, add_array, size, x_array);
}

HWY_EXPORT(LogLoop);

void CallLogLoop(const float* in_array, float* out_array, const size_t size) {
    // This must reside outside of HWY_NAMESPACE because it references (calls
    // the appropriate one from) the per-target implementations there. For
    // static dispatch, use HWY_STATIC_DISPATCH.
    return HWY_DYNAMIC_DISPATCH(LogLoop)(in_array, out_array, size);
}

void MulAddVector(const float* mul_array, const float* add_array, const size_t size, float* x_array) {
    // std::cerr << __PRETTY_FUNCTION__ << std::endl;
#pragma clang loop vectorize(enable) interleave(enable)
    for (size_t i = 0; i < size; ++i) {
        x_array[i] = mul_array[i] * x_array[i] + add_array[i];
    }
}

void MulAddScalar(const float* mul_array, const float* add_array, const size_t size, float* x_array) {
    // std::cerr << __PRETTY_FUNCTION__ << std::endl;
#pragma clang loop vectorize(disable) interleave(disable)
    for (size_t i = 0; i < size; ++i) {
        x_array[i] = mul_array[i] * x_array[i] + add_array[i];
    }
}

// This macro declares a static array used for dynamic dispatch.
// HWY_EXPORT(LogLoop);

// void CallMulAddLoop(const float*  in_array,
//                 const float*  out_array,
//                 const size_t size) {
//   // This must reside outside of HWY_NAMESPACE because it references (calls
//   the
//   // appropriate one from) the per-target implementations there.
//   // For static dispatch, use HWY_STATIC_DISPATCH.
//   return HWY_DYNAMIC_DISPATCH(MulAddLoop)(in_array, out_array, size);
// }

}  // namespace project

namespace {

static constexpr auto size = 1 << 22;
static constexpr auto iterations = 1000;
static constexpr auto threads = 1;

alignas(32) std::array<float, size> mul_array;
alignas(32) std::array<float, size> add_array;
alignas(32) std::array<float, size> x_array;
alignas(32) std::array<float, size> log_in_array;
alignas(32) std::array<float, size> log_out_array;
alignas(32) std::array<float, size> log_reference_array;

void HwyVector(benchmark::State& state) {
    for (auto _ : state) {
        project::CallMulAddLoop(mul_array.data(), add_array.data(), size, x_array.data());
    }
}

void Scalar(benchmark::State& state) {
    for (auto _ : state) {
        project::MulAddScalar(mul_array.data(), add_array.data(), size, x_array.data());
    }
}

void Vector(benchmark::State& state) {
    for (auto _ : state) {
        project::MulAddVector(mul_array.data(), add_array.data(), size, x_array.data());
    }
}

/* compute natural logarithm, maximum error 0.85089 ulps \
 * source https://stackoverflow.com/questions/39821367/very-fast-approximate-logarithm-natural-log-function-in-c
 */
inline constexpr float fast_logf(float a) {
    float i, m, r, s, t;
    int e;
    // print a
    i = 0.0f;
    if (a < 1.175494351e-38f) {  // 0x1.0p-126
        a = a * 8388608.0f;      // 0x1.0p+23
        i = -23.0f;
    }
    e = (std::bit_cast<int32_t>(a) - std::bit_cast<int32_t>(0.666666667f)) & 0xff800000;

    m = std::bit_cast<float>(std::bit_cast<int32_t>(a) - e);
    i = std::fma((float)e, 1.19209290e-7f, i);  // 0x1.0p-23

    /* m in [2/3, 4/3] */
    m = m - 1.0f;
    s = m * m;

    /* Compute log1p(m) for m in [-1/3, 1/3] */
    r = -0.130310059f;  // -0x1.0ae000p-3

    t = 0.140869141f;                   //  0x1.208000p-3
    r = std::fma(r, s, -0.121483512f);  // -0x1.f198b2p-4
    t = std::fma(t, s, 0.139814854f);   //  0x1.1e5740p-3
    r = std::fma(r, s, -0.166846126f);  // -0x1.55b36cp-3
    t = std::fma(t, s, 0.200120345f);   //  0x1.99d8b2p-3
    r = std::fma(r, s, -0.249996200f);  // -0x1.fffe02p-3
    r = std::fma(t, m, r);
    r = std::fma(r, m, 0.333331972f);   //  0x1.5554fap-2
    r = std::fma(r, m, -0.500000000f);  // -0x1.000000p-1
    r = std::fma(r, s, m);
    r = std::fma(i, 0.693147182f, r);  //  0x1.62e430p-1 // log(2)
    if (!((a > 0.0f) && (a < INFINITY))) {
        r = a + a;                              // silence NaNs if necessary
        if (a < 0.0f) r = INFINITY - INFINITY;  //  NaN
        if (a == 0.0f) r = -INFINITY;
    }
    return r;
}

void LogScalar(benchmark::State& state) {
    for (auto _ : state) {
#pragma clang loop vectorize(disable) interleave(disable)
        for (size_t i = 0; i < size; ++i) {
            log_out_array[i] = fast_logf(log_in_array[i]);
        }
    }
}

void LogVector(benchmark::State& state) {
    for (auto _ : state) {
#pragma clang loop vectorize(enable) interleave(enable)
        for (size_t i = 0; i < size; ++i) {
            log_reference_array[i] = fast_logf(log_in_array[i]);
        }
    }
}

void StdLogScalar(benchmark::State& state) {
    for (auto _ : state) {
#pragma clang loop vectorize(disable) interleave(disable)
        for (size_t i = 0; i < size; ++i) {
            log_out_array[i] = std::log(log_in_array[i]);
        }
    }
}

void StdLogVector(benchmark::State& state) {
    for (auto _ : state) {
#pragma clang loop vectorize(enable) interleave(enable)
        for (size_t i = 0; i < size; ++i) {
            log_out_array[i] = std::log(log_in_array[i]);
        }
    }
}

void LogHwy(benchmark::State& state) {
    for (auto _ : state) {
        project::CallLogLoop(log_in_array.data(), log_out_array.data(), size);
    }
}

// this function checks the error of the log function comparing log_in_array and log_reference_array
// it uses relative error
void CheckLogError() {
    float max_error = 0.0f;
    float max_relative_error = 0.0f;
    for (size_t i = 0; i < size; ++i) {
        float error = std::abs(log_out_array[i] - log_reference_array[i]);
        float relative_error = error / std::abs(log_reference_array[i]);
        if (error > max_error) {
            max_error = error;
        }
        if (relative_error > max_relative_error) {
            max_relative_error = relative_error;
        }
    }
    std::cerr << "max error: " << max_error << std::endl;
    std::cerr << "max relative error: " << max_relative_error << std::endl;
}

BENCHMARK(Scalar)->Iterations(iterations)->Threads(threads);
BENCHMARK(Vector)->Iterations(iterations)->Threads(threads);
BENCHMARK(HwyVector)->Iterations(iterations)->Threads(threads);
BENCHMARK(StdLogScalar)->Iterations(iterations)->Threads(threads);
BENCHMARK(StdLogVector)->Iterations(iterations)->Threads(threads);
BENCHMARK(LogScalar)->Iterations(iterations)->Threads(threads);
BENCHMARK(LogVector)->Iterations(iterations)->Threads(threads);
BENCHMARK(LogHwy)->Iterations(iterations)->Threads(threads);

}  // namespace

int main(int argc, char** argv) {
    std::random_device rd;
    std::uniform_real_distribution<float> dist{0.0, 1.0};
    std::uniform_real_distribution<float> log_dis{0.f, std::numeric_limits<float>::max()};

    for (size_t j = 0; j < 1; j++) {
        for (size_t i = 0; i < size; ++i) {
            mul_array[i] = dist(rd);
            add_array[i] = dist(rd);
            x_array[i] = dist(rd);
            log_in_array[i] = log_dis(rd);
        }
    }
    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    CheckLogError();
    return 0;
}

#endif  // HWY_ONCE
