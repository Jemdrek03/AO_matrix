#include <cstdint>
#include <iostream>
#include <ctime>
#include <thread>
#include <vector>
#include <chrono>


#define N 2048
#define timeNumber 1



uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}


alignas(64) float A[N][N];
alignas(64) float B[N][N];
alignas(64) float C[N][N];


// ONLY for square matrices for now!!!
void threading(int rowNumber)
{
    for(int x = 0; x < N; x++)
    {
        float acc = 0;
        for(int k = 0; k < N; k++)
        {
            acc += A[rowNumber][k] * B[k][x];
        }
        C[rowNumber][x] = acc;
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
        for (int t = 0; t < N; t++) {
            th.emplace_back(threading, t);
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