//
// Created by mbarb on 6/17/2020.
//
#include <bitset>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

typedef union {
  float f;
  struct {
    unsigned int mantissa : 23;
    unsigned int exponent : 8;
    unsigned int sign : 1;
  } parts;
} float_cast;

template <typename T> static std::string toBinaryExponent(const T &x) {
  std::stringstream ss;
  ss << std::bitset<8>(x);
  return ss.str();
}

template <typename T> static std::string toBinaryMantissa(const T &x) {
  std::stringstream ss;
  ss << 1 << std::bitset<23>(x);
  return ss.str();
}

static std::string toBinFloat(const float &x) {
  std::stringstream ss;
  float_cast d1 = {.f = x};
  ss << std::bitset<1>(d1.parts.sign) << '|';
  ss << std::bitset<8>(d1.parts.exponent - 127) << '|';
  ss << std::bitset<23>(d1.parts.mantissa);
  return ss.str();
}

static std::string toIntFloat(const float &x) {
  std::stringstream ss;
  float_cast d1 = {.f = x};
  ss << d1.parts.sign << '|' << std::setfill('0') << std::setw(3)
     << d1.parts.exponent << '|' << std::setfill('0') << std::setw(7)
     << d1.parts.mantissa;
  return ss.str();
}

// trying to find a pattern for a floating point indexed table
// const auto n = 2;
const auto a = 0;
const auto b = 1;

using namespace std;

int main(const int argc, const char *argv[]) {
  for (auto n = 1u; n < 8192u; n <<= 1u) {
    const float value = 1. / n;
    cout << n << '\t' << toBinFloat(value) << '\t' << toIntFloat(value)
         << '\t' << value << endl;
  }

  for (auto n = 1u; n < 4094u; n <<= 1u) {
    cout << "n " << n << endl;
    const float h = 1. / n;
    //  cout << h << endl;
    map<unsigned int, unsigned int> exponents;
    map<unsigned int, unsigned int> mantissa;
    for (auto i = 0; i < n; ++i) {
      const float value = (a + h * i)*n;
      float_cast d1 = {.f = value};
      const auto exp = d1.parts.exponent;
      mantissa[d1.parts.mantissa]++;
      exponents[exp]++;
      cout << i << '\t' << toBinFloat(value) << '\t' << toIntFloat(value)
           << '\t' << value << endl;
    }
    //  for (int i = 0; i < n; ++i) {
    //    if (exponents[i] != 0)
    //      cout << i << '\t' << exponents[i] << endl;
    //  }
    //  for(auto[key, value] : mantissa){
    //      cout << key << '\t' << value << endl;
    //  }
    cout << "mantissa " << mantissa.size() << endl;
    cout << "exponents " << exponents.size() << endl;
  }
  return 0;
}