cmake_minimum_required(VERSION 3.28)
project(GPU LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(CUDAToolkit REQUIRED)


add_executable(GPU
        #GPU_Threading.cpp
        #CUDA_Version1.cu
        #CUDA_Version2.cu
        CUDA_Version3.cu
        #CUDA_testing_CUBLAs.cu

)
target_compile_options(GPU PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>)
target_link_libraries(GPU PRIVATE
        CUDA::cudart
        CUDA::cublas)
