#include <cstdint>
#include <iostream>
#include <ctime>
#include <cassert>
#include <chrono>

#define N 1024
#define BLOCK 32
#define timeNumber 10.0

uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}
float A[N][N];
float B[N][N];
float C[N][N];



int main()
{
    uint64_t startone = nanos();
    assert(N%BLOCK == 0);
    double gflop = (N * N * 2.0 * N) * 1e-9;
    double sumTime = 0.0;

    for( int t = 0; t < timeNumber; t++)
    {
        uint64_t start = nanos();
        for (int by = 0; by < N; by += BLOCK)
            for (int bx = 0; bx < N; bx += BLOCK) {


                for (int y = by; y < by + BLOCK; ++y)
                    for (int x = bx; x < bx + BLOCK; ++x)
                        C[y][x] = 0.0f;

                for (int bk = 0; bk < N; bk += BLOCK)
                    for (int y = by; y < by + BLOCK; ++y)
                        for (int k = bk; k < bk + BLOCK; ++k) {
                            float a = A[y][k];
                            for (int x = bx; x < bx + BLOCK; ++x)
                                C[y][x] += a * B[k][x];
                        }
            }

        uint64_t end = nanos();
        double s = (end - start) * 1e-9;
        sumTime += s;
    }

    uint64_t endone = nanos();
    //std::cout << " Time " << (endone - startone) * 1e-9 << " seconds" <<std::endl;
    std::cout << " Average latency " << (sumTime / timeNumber)<< " seconds " <<std::endl;
    std::cout << " Average GFLOPS " << gflop / (sumTime / timeNumber)<<std::endl;
    return 0;
}