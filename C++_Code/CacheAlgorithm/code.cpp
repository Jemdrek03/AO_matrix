#include <cstdint>
#include <iostream>
#include <ctime>
#include <cassert>

#define N 2048

uint64_t nanos(){
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}
float A[N][N];
float B[N][N];
float C[N][N];


#define BLOCK 32
int main()
{
    assert(N%BLOCK == 0);

    uint64_t start = nanos();
    for(int by = 0; by < N;by += BLOCK){
        for(int bx = 0; bx < N; bx += BLOCK){


            for(int y = by; y < by + BLOCK;y++) {
                for (int x = bx; x < bx + BLOCK; x++) {
                    float acc = 0;
                    for (int k = 0; k < N; k++) {
                        acc += A[y][k] * B[k][x];
                    }
                    C[y][x] = acc;
                }
            }



        }
    }
    uint64_t end = nanos();
    double gflop = (N*N*2.0*N)*1e-9;
    double s = (end-start)*1e-9;
    std::cout<<"FLOPS "<<gflop/s;
    return 0;
}