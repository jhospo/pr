cmake_minimum_required(VERSION 3.27)
project(prlab1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

add_executable(prlab1 main.cpp)
