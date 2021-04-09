//
// Created by mbarb on 2/19/2021.
//

#include <iostream>
using namespace std;
#include "print_const.h"
#undef CPP_LEARNING_COMPILATION_DEFINE_TEST_PRINT_CONST_H_
#define VALUE 42
#include "print_const.h"

int main() {
    print_const();
    print_const_value();
    return 0;
}