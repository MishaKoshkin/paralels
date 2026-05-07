cmake_minimum_required(VERSION 3.10)

project(SinTaylorSum CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_EXTENSIONS OFF)

option(USE_FLOAT "Использовать float вместо double для типа type" OFF)

add_executable(sin_taylor_sum sin.cpp)

if(USE_FLOAT)
    target_compile_definitions(myapp PRIVATE USE_FLOAT)
    message(STATUS "Сборка с float precision (USE_FLOAT=ON)")
else()
    message(STATUS "Сборка с double precision (USE_FLOAT=OFF)")
endif()

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")