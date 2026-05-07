cmake_minimum_required(VERSION 3.10)
project(ClientServer)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)

add_executable(client_server main.cpp)
add_executable(test test.cpp)

target_link_libraries(client_server Threads::Threads)
target_link_libraries(test Threads::Threads)