// nary_dispatch.hpp
#pragma once

#include <array>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

// ---------- helpers ----------
template <int Offset, typename Seq>
struct offset_seq;
template <int Offset, int... I>
struct offset_seq<Offset, std::integer_sequence<int, I...>> {
    using type = std::integer_sequence<int, (Offset + I)...>;
};
template <int Start, int End>
using make_range = typename offset_seq<Start, std::make_integer_sequence<int, End - Start + 1>>::type;

template <typename Seq>
struct DispatchParam {
    int runtime_val;
    using seq_type = Seq;
};

struct FuncKernel {
    template <int... P>
    int operator()(int x) const noexcept {
        return (0 + ... + P) + x;
    }
};

// ---------- sequence traits ----------
template <typename Seq>
struct seq_info;

template <int First, int... Rest>
struct seq_info<std::integer_sequence<int, First, Rest...>> {
    static constexpr int len = 1 + sizeof...(Rest);

    static constexpr std::array<int, len> values() noexcept { return std::array<int, len>{First, Rest...}; }
    static constexpr int min_value() noexcept {
        constexpr auto v = values();
        int m = v[0];
        for (int i = 1; i < len; ++i)
            if (v[i] < m) m = v[i];
        return m;
    }
    static constexpr int max_value() noexcept {
        constexpr auto v = values();
        int M = v[0];
        for (int i = 1; i < len; ++i)
            if (v[i] > M) M = v[i];
        return M;
    }
    static constexpr int span() noexcept { return max_value() - min_value() + 1; }

    // value -> index in values(); O(1) if span small, else O(len)
    static int map_to_index(int v) {
        constexpr int DenseSpanLimit = 64;
        if constexpr (span() <= DenseSpanLimit) {
            static const std::array<int, span()> lut = [] {
                std::array<int, span()> a{};
                for (int i = 0; i < span(); ++i) a[i] = -1;
                constexpr auto vals = values();
                for (int i = 0; i < len; ++i) a[vals[i] - min_value()] = i;
                return a;
            }();
            const int off = v - min_value();
            if (off < 0 || off >= span()) return -1;
            return lut[off];
        } else {
            constexpr auto vals = values();
            for (int i = 0; i < len; ++i)
                if (vals[i] == v) return i;
            return -1;
        }
    }
};

template <typename Seq, std::size_t K>
struct seq_at;
template <int H, int... T>
struct seq_at<std::integer_sequence<int, H, T...>, 0> : std::integral_constant<int, H> {};
template <int H, int... T, std::size_t K>
struct seq_at<std::integer_sequence<int, H, T...>, K> : seq_at<std::integer_sequence<int, T...>, K - 1> {};

template <typename A, typename B>
struct seq_cat;
template <int... X, int... Y>
struct seq_cat<std::integer_sequence<int, X...>, std::integer_sequence<int, Y...>> {
    using type = std::integer_sequence<int, X..., Y...>;
};

template <typename Tuple>
struct seq_tuple;
template <typename... P>
struct seq_tuple<std::tuple<P...>> {
    using type = std::tuple<typename P::seq_type...>;
};
template <typename Tuple>
using seq_tuple_t = typename seq_tuple<std::decay_t<Tuple>>::type;

template <typename Func, typename ArgTuple, typename... Seq>
struct result_type_helper {
    template <std::size_t... I>
    static auto test(std::index_sequence<I...>)
        -> decltype(std::declval<Func>().template operator()<seq_at<Seq, 0>::value...>(
            std::get<I>(std::declval<ArgTuple>())...));
    using type = decltype(test(std::make_index_sequence<std::tuple_size<ArgTuple>::value>{}));
};
template <typename Func, typename ArgTuple, typename SeqTuple>
struct result_type;
template <typename Func, typename ArgTuple, typename... Seq>
struct result_type<Func, ArgTuple, std::tuple<Seq...>> {
    using type = typename result_type_helper<Func, ArgTuple, Seq...>::type;
};
template <typename Func, typename ArgTuple, typename SeqTuple>
using result_t = typename result_type<Func, ArgTuple, SeqTuple>::type;

// ---------- linear-scan (baseline) ----------
namespace linear_detail {

template <typename F, typename... Seq>
struct Product;
template <typename F, int... I1, typename Seq2, typename... Rest>
struct Product<F, std::integer_sequence<int, I1...>, Seq2, Rest...> {
    template <int... Prefix>
    static void apply(F& f) {
        (Product<F, Seq2, Rest...>::template apply<Prefix..., I1>(f), ...);
    }
};
template <typename F, int... I1>
struct Product<F, std::integer_sequence<int, I1...>> {
    template <int... Prefix>
    static void apply(F& f) {
        (f.template operator()<Prefix..., I1>(), ...);
    }
};
template <typename F, typename... Seq>
inline void product(F& f, Seq...) {
    Product<F, Seq...>::template apply<>(f);
}

template <typename Tuple, std::size_t... I>
inline auto extract_vals_impl(const Tuple& t, std::index_sequence<I...>) {
    return std::array<int, sizeof...(I)>{std::get<I>(t).runtime_val...};
}
template <typename Tuple>
inline auto extract_vals(const Tuple& t) {
    using T = std::remove_reference_t<Tuple>;
    return extract_vals_impl(t, std::make_index_sequence<std::tuple_size_v<T>>{});
}
template <typename Tuple, std::size_t... I>
inline auto extract_seqs_impl(const Tuple&, std::index_sequence<I...>) {
    using T = std::remove_reference_t<Tuple>;
    return std::tuple<typename std::tuple_element_t<I, T>::seq_type...>{};
}
template <typename Tuple>
inline auto extract_seqs(const Tuple& t) {
    using T = std::remove_reference_t<Tuple>;
    return extract_seqs_impl(t, std::make_index_sequence<std::tuple_size_v<T>>{});
}

template <typename Func, typename ArgTuple, typename... Seq>
struct dispatch_result_helper {
    template <std::size_t... I>
    static auto test(std::index_sequence<I...>)
        -> decltype(std::declval<Func>().template operator()<seq_at<Seq, 0>::value...>(
            std::get<I>(std::declval<ArgTuple>())...));
    using type = decltype(test(std::make_index_sequence<std::tuple_size_v<ArgTuple>>{}));
};
template <typename Func, typename ArgTuple, typename SeqTuple>
struct dispatch_result;
template <typename Func, typename ArgTuple, typename... Seq>
struct dispatch_result<Func, ArgTuple, std::tuple<Seq...>> {
    using type = typename dispatch_result_helper<Func, ArgTuple, Seq...>::type;
};
template <typename Func, typename ArgTuple, typename SeqTuple>
using dispatch_result_t = typename dispatch_result<Func, ArgTuple, SeqTuple>::type;

template <typename Func, std::size_t N, typename ArgTuple, typename ResultType>
struct Caller {
    Func& func;
    const std::array<int, N>& vals;
    ArgTuple& args;
    std::conditional_t<std::is_void_v<ResultType>, char, ResultType> result{};
    template <int... Params>
    void operator()() {
        static constexpr std::array<int, sizeof...(Params)> p{Params...};
        if (p == vals) {
            if constexpr (std::is_void_v<ResultType>) {
                std::apply([&](auto&&... a) { func.template operator()<Params...>(std::forward<decltype(a)>(a)...); },
                           args);
            } else {
                result = std::apply(
                    [&](auto&&... a) { return func.template operator()<Params...>(std::forward<decltype(a)>(a)...); },
                    args);
            }
        }
    }
};

}  // namespace linear_detail

template <typename Func, typename ParamTuple, typename... Args>
[[nodiscard]] decltype(auto) linear_dispatch(Func&& func, ParamTuple&& params, Args&&... args) {
    auto vals = linear_detail::extract_vals(params);
    auto seqs = linear_detail::extract_seqs(params);
    auto arg_tuple = std::forward_as_tuple(std::forward<Args>(args)...);
    using result_t_lin = linear_detail::dispatch_result_t<Func, decltype(arg_tuple), decltype(seqs)>;
    linear_detail::Caller<Func, std::tuple_size_v<std::decay_t<ParamTuple>>, decltype(arg_tuple), result_t_lin> caller{
        func, vals, arg_tuple};
    std::apply([&](auto&&... s) { linear_detail::product(caller, s...); }, seqs);
    if constexpr (!std::is_void_v<result_t_lin>) return caller.result;
}

// ---------- variant + visit ----------
namespace visit_detail {

template <int... I>
struct variant_maker {
    using V = std::variant<std::integral_constant<int, I>...>;
    using Fn = V (*)();
    static V make_by_index(int idx) {
        static constexpr Fn tbl[] = {+[]() -> V { return V{std::integral_constant<int, I>{}}; }...};
        return tbl[idx]();
    }
};

template <typename Seq>
struct variant_from_seq;
template <int... I>
struct variant_from_seq<std::integer_sequence<int, I...>> {
    using type = typename variant_maker<I...>::V;
    static std::optional<type> make_from_value(int v) {
        const int idx = seq_info<std::integer_sequence<int, I...>>::map_to_index(v);
        if (idx < 0) return std::nullopt;
        return variant_maker<I...>::make_by_index(idx);
    }
};
template <typename Seq>
using variant_from_seq_t = typename variant_from_seq<Seq>::type;

template <typename ParamTuple, std::size_t... K>
inline auto make_variants(const ParamTuple& params, std::index_sequence<K...>) {
    using P = std::decay_t<ParamTuple>;
    return std::make_tuple(variant_from_seq<typename std::tuple_element_t<K, P>::seq_type>::make_from_value(
        std::get<K>(params).runtime_val)...);
}

template <typename Func, typename ArgTuple>
struct Visitor {
    Func* func;
    ArgTuple* args;
    template <typename... IC>
    auto operator()(IC...) -> decltype(auto) {
        return std::apply(
            [this](auto&&... a) -> decltype(auto) {
                return this->func->template operator()<IC::value...>(std::forward<decltype(a)>(a)...);
            },
            *args);
    }
};

}  // namespace visit_detail

template <typename Func, typename ParamTuple, typename... Args>
[[nodiscard]] decltype(auto) visit_dispatch(Func& func, const ParamTuple& params, Args&&... args) {
    using SeqTuple = seq_tuple_t<ParamTuple>;
    using R = result_t<Func, std::tuple<Args...>, SeqTuple>;
    auto arg_tuple = std::forward_as_tuple(std::forward<Args>(args)...);

    auto variants = visit_detail::make_variants(
        params, std::make_index_sequence<std::tuple_size<std::decay_t<ParamTuple>>::value>{});

    bool ok = true;
    std::apply([&](auto const&... vo) { (void)std::initializer_list<int>{(ok = ok && vo.has_value(), 0)...}; },
               variants);
    if (!ok) {
        if constexpr (!std::is_void_v<R>)
            return R{};
        else
            return;
    }

    visit_detail::Visitor<std::decay_t<Func>, decltype(arg_tuple)> vis{&func, &arg_tuple};
    if constexpr (std::is_void_v<R>) {
        std::apply([&](auto&... vo) { std::visit(vis, *vo...); }, variants);
    } else {
        return std::apply([&](auto&... vo) { return std::visit(vis, *vo...); }, variants);
    }
}

// ---------- jump-table ----------
namespace table_detail {

// flat-index -> integer_sequence<int, Vals...>
template <std::size_t I, typename... Seq>
struct flat_to_vals;
template <std::size_t I>
struct flat_to_vals<I> {
    using type = std::integer_sequence<int>;
};
template <std::size_t I, typename Seq1, typename... Rest>
struct flat_to_vals<I, Seq1, Rest...> {
    static constexpr int len1 = seq_info<Seq1>::len;
    static constexpr std::size_t d1 = I % static_cast<std::size_t>(len1);
    static constexpr std::size_t next = I / static_cast<std::size_t>(len1);
    using head = std::integer_sequence<int, seq_at<Seq1, d1>::value>;
    using tail = typename flat_to_vals<next, Rest...>::type;
    using type = typename seq_cat<head, tail>::type;
};

template <typename Func, typename R, typename SeqVals, typename... CallArgs>
struct EntryPtr;
template <typename Func, typename R, int... Vals, typename... CallArgs>
struct EntryPtr<Func, R, std::integer_sequence<int, Vals...>, CallArgs...> {
    static R call(Func* f, CallArgs... a) {
        if constexpr (std::is_void_v<R>) {
            f->template operator()<Vals...>(std::forward<CallArgs>(a)...);
        } else {
            return f->template operator()<Vals...>(std::forward<CallArgs>(a)...);
        }
    }
};

template <typename Func, typename SeqTuple, typename... Args>
struct Engine;
template <typename Func, typename... Seq, typename... Args>
struct Engine<Func, std::tuple<Seq...>, Args...> {
    using R = result_t<Func, std::tuple<Args...>, std::tuple<Seq...>>;
    using Fn = std::conditional_t<std::is_void_v<R>, void (*)(Func*, Args...), R (*)(Func*, Args...)>;

    static constexpr int total() noexcept { return (seq_info<Seq>::len * ...); }

    template <std::size_t... I>
    static auto make_table(std::index_sequence<I...>) {
        return std::array<Fn, static_cast<std::size_t>(total())>{
            &EntryPtr<Func, std::conditional_t<std::is_void_v<R>, void, R>, typename flat_to_vals<I, Seq...>::type,
                      Args...>::call...};
    }

    static const std::array<Fn, static_cast<std::size_t>(total())>& table() {
        static const auto tbl = make_table(std::make_index_sequence<static_cast<std::size_t>(total())>{});
        return tbl;
    }

    template <typename ParamTuple>
    static auto dispatch(Func& func, const ParamTuple& params, Args&&... args)
        -> std::conditional_t<std::is_void_v<R>, void, R> {
        int idx = 0, stride = 1;
        bool ok = true;
        expand_dispatch(std::make_index_sequence<sizeof...(Seq)>{}, idx, stride, ok, params);
        if (!ok) {
            if constexpr (!std::is_void_v<R>)
                return R{};
            else
                return;
        }
        const auto& tbl = table();
        if constexpr (std::is_void_v<R>) {
            tbl[static_cast<std::size_t>(idx)](&func, std::forward<Args>(args)...);
        } else {
            return tbl[static_cast<std::size_t>(idx)](&func, std::forward<Args>(args)...);
        }
    }

   private:
    template <std::size_t... K, typename ParamTuple>
    static void expand_dispatch(std::index_sequence<K...>, int& idx, int& stride, bool& ok, const ParamTuple& params) {
        (void)std::initializer_list<int>{(
            [&] {
                using S = std::tuple_element_t<K, std::tuple<Seq...>>;
                const int off = seq_info<S>::map_to_index(std::get<K>(params).runtime_val);
                ok &= (off >= 0);
                idx += off * stride;
                stride *= seq_info<S>::len;
            }(),
            0)...};
    }
};

}  // namespace table_detail

template <typename Func, typename ParamTuple, typename... Args>
[[nodiscard]] decltype(auto) table_dispatch(Func& func, const ParamTuple& params, Args&&... args) {
    using SeqTuple = seq_tuple_t<ParamTuple>;
    return table_detail::Engine<Func, SeqTuple, Args...>::dispatch(func, params, std::forward<Args>(args)...);
}
