#include <cstdint>
#include <iostream>
#include <ctime>
#include <cassert>
#include <chrono>

#define N 2048

uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}
float A[N][N];
float B[N][N];
float C[N][N];


#define BLOCK 32
int main()
{
    assert(N%BLOCK == 0);

    uint64_t start = nanos();

    for (int by=0; by<N; by+=BLOCK)
        for (int bx=0; bx<N; bx+=BLOCK) {


            for (int y=by; y<by+BLOCK; ++y)
                for (int x=bx; x<bx+BLOCK; ++x)
                    C[y][x] = 0.0f;

            for (int bk=0; bk<N; bk+=BLOCK)
                for (int y=by; y<by+BLOCK; ++y)
                    for (int k=bk; k<bk+BLOCK; ++k) {
                        float a = A[y][k];
                        for (int x=bx; x<bx+BLOCK; ++x)
                            C[y][x] += a * B[k][x];
                    }
        }

    uint64_t end = nanos();
    double gflop = (N*N*2.0*N)*1e-9;
    double s = (end-start)*1e-9;
    std::cout<<"GFLOPS "<<gflop/s;
    return 0;
}