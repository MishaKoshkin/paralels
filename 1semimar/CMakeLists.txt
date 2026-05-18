cmake_minimum_required(VERSION 3.10)
project(sin_sum)

option(USE_DOUBLE "Использовать double вместо float" OFF)

if(USE_DOUBLE)
    add_definitions(-DUSE_DOUBLE)
endif()

add_executable(sin_sum sin.cpp)