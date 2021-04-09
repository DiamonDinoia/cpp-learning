//
// Created by mbarb on 2/19/2021.
//

#ifndef CPP_LEARNING_COMPILATION_DEFINE_TEST_PRINT_CONST_H_
#define CPP_LEARNING_COMPILATION_DEFINE_TEST_PRINT_CONST_H_

#ifndef VALUE
void print_const() { cout << "no const" << endl; }
#else
void print_const_value() { cout << VALUE << endl; }
#endif
#endif  // CPP_LEARNING_COMPILATION_DEFINE_TEST_PRINT_CONST_H_
