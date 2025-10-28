#include <cstdint>
#include <iostream>
#include <ctime>
#include <thread>
#include <vector>
#include <algorithm>


#define N 2048
#define blocksize 16
#define timeNumber 1
uint64_t nanos(){
    struct timespec start;
    clock_gettime(CLOCK_MONOTONIC, &start);
    return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}


alignas(64) float A[N][N];
alignas(64) float B[N][N];
alignas(64) float C[N][N];


// ONLY for square matrices for now!!!
void threading(int rowNumber, int rowNumber2)
{
    rowNumber2 = std::min(rowNumber2,N); // So it doesnt leave the N sized matrix
    for(int y = rowNumber; y < rowNumber2; y++) {
        for (int x = 0; x < N; x++) {
            float acc = 0;
            for (int k = 0; k < N; k++) {
                acc += A[y][k] * B[k][x];
            }
            C[y][x] = acc;
        }
    }

}


int main()
{

    // Matrix initialization
    for (int y=0; y<N; ++y)
        for (int x=0; x<N; ++x) {
            A[y][x] = (y + x) * 0.001f;
            B[y][x] = (y - x) * 0.002f;
        }

    for(int times = 0; times < timeNumber; times++) {
        // Start to count the time needed
        uint64_t start = nanos();


        // Whole threading logic
        std::vector<std::thread> th;
        for (int t = 0; t < N; t += blocksize) {
            int t1 = t + blocksize;
            th.emplace_back(threading, t, t1);
        }
        for (auto &t: th)
            t.join();


        // Finalize time counting
        uint64_t end = nanos();
        double gflop = (N * N * 2.0 * N) * 1e-9;
        double s = (end - start) * 1e-9;
        std::cout << "GFLOPS " << gflop / s<<std::endl;
    }

    return 0;
}