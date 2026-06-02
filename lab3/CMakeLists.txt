cmake_minimum_required(VERSION 3.10)
project(lab3)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

# Цели из папки task2
add_executable(client_server task2/main.cpp)
add_executable(test task2/test.cpp)

target_link_libraries(client_server Threads::Threads)
target_link_libraries(test Threads::Threads)

# Цель из папки task1
add_executable(vector-matrix task1/main.cpp)
target_link_libraries(vector-matrix Threads::Threads)