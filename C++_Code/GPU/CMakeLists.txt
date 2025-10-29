cmake_minimum_required(VERSION 3.28)
project(GPU LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(GPU
        #GPU_Threading.cpp
        CUDA_Version1.cu

)
