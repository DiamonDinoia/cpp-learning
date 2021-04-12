//
// Created by mbarbone on 15/07/2020.
//

#include <unistd.h>

#include <chrono>
#include <iostream>
#include <thread>

using namespace std;

void task1(std::string msg) { std::cout << "task1 says: " << msg; }

void clone_flags() {
    auto pid = fork();
    if (pid == 0) {
        exit(0);
    }
    std::thread t1(task1, "Hello ");
    t1.join();
    std::cout << "exiting" << std::endl;
}

constexpr unsigned long iterations = 1 << 21;
// from:
// https://stackoverflow.com/questions/3929774/how-much-overhead-is-there-when-creating-a-thread
void pthread_benchmark() {
    cout << "testing thread creation on " << iterations << " iterations"
         << endl;
    const auto start = std::chrono::high_resolution_clock::now();
    for (auto i = 0U; i < iterations; i++) {
        std::thread([]() {}).detach();
    }
    const auto end = std::chrono::high_resolution_clock::now();

    const std::chrono::duration<double, std::micro> duration = end - start;
    cout << "average thread overhead "
         << duration.count() / static_cast<double>(iterations)
         << " micro-seconds" << endl;
}

int main() {
    clone_flags();
    pthread_benchmark();
}