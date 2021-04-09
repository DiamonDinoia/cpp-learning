//
// Created by mbarbone on 15/07/2020.
//

#include <unistd.h>

#include <iostream>
#include <thread>

void task1(std::string msg) { std::cout << "task1 says: " << msg; }

int main(const int argc, const char *argv[]) {
    std::thread t1(task1, "Hello");
    fork();
    t1.join();
    std::cout << "exiting " << std::endl;
}