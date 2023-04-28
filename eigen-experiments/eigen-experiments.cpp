//
// Created by mbarbone on 12/2/21.
//
#include <Eigen/Dense>
#include <iostream>
template <typename T>
class ThreeVector : public Eigen::Array<T, 3, 1> {
   public:
    ThreeVector(T &&x, T &&y, T &&z) : Eigen::Array<T, 3, 1>(x, y, z) {}
    ThreeVector(const T &x, const T &y, const T &z)
        : Eigen::Array<T, 3, 1>(x, y, z) {}
    ThreeVector(const Eigen::Array<T, 3, 1> &other) : Eigen::Array<T, 3, 1>(other) {}
    ThreeVector(Eigen::Array<T, 3, 1> &&other) noexcept : Eigen::Array<T, 3, 1>(other) {}
    ThreeVector(const ThreeVector &other) : Eigen::Array<T, 3, 1>(other) {}
    ThreeVector(ThreeVector &&other) noexcept : Eigen::Array<T, 3, 1>(other) {}
    T &x = Eigen::Array<T, 3, 1>::operator[](0);
    T &y = Eigen::Array<T, 3, 1>::operator[](1);
    T &z = Eigen::Array<T, 3, 1>::operator[](2);
    friend std::ostream &operator<<(std::ostream &os,
                                    const ThreeVector &vector) {
        os << "[ x: " << vector.x << " y: " << vector.y << " z: " << vector.z
           << "]";
        return os;
    }

};

int main(const int argc, const char *argv[]) {
    ThreeVector<double> test{1, 1, 1};
    ThreeVector<double> test2{0.5, 0.5, 0.5};
    for (int i = 0; i < 1024; ++i) {
        test += test2 * test * test;
    }
    std::cout << test << std::endl;
    //    std::cout << test.x << std::endl;
    //    test.x += 3;
    //    std::cout << test.x << std::endl;

    return 0;
}