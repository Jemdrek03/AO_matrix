#include <cstdint>
#include <iostream>
#include <ctime>
#include <cassert>
#include <atomic>
#include <chrono>

#define N 2048
#define timeNumber 10.0
#define BLOCK 32


static float A[N][N], B[N][N], C[N][N], BT[N][N];

uint64_t nanos() {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()
    ).count();
}

int main() {

    uint64_t startone = nanos();
    assert(N%BLOCK == 0);
    double gflop = (N * N * 2.0 * N) * 1e-9;
    double sumTime = 0.0;

    // Matrix initialization
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) {
            A[i][j] = (i + j) * 0.001f;
            B[i][j] = ((i*37 + j*17) % 101) * 0.01f;
        }

    // Transposition
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            BT[j][i] = B[i][j];



    for(int t =0; t < timeNumber; t++) {
        //std::atomic_thread_fence(std::memory_order_seq_cst);
        uint64_t start = nanos();

        for (int by = 0; by < N; by += BLOCK)
            for (int bx = 0; bx < N; bx += BLOCK)
                for (int y = by; y < by + BLOCK; ++y)
                    for (int x = bx; x < bx + BLOCK; ++x) {
                        float acc = 0.0f;
                        for (int bk = 0; bk < N; bk += BLOCK)
                            for (int k = bk; k < bk + BLOCK; ++k)
                                acc += A[y][k] * BT[x][k];
                        C[y][x] = acc;
                    }

        //std::atomic_thread_fence(std::memory_order_seq_cst);
        uint64_t end = nanos();
        double s = (end - start) * 1e-9;
        sumTime += s;
    }

    uint64_t endone = nanos();
    //std::cout << " Time " << (endone - startone) * 1e-9 << " seconds" <<std::endl;
    std::cout << " Average latency " << (sumTime / timeNumber)<< " seconds " <<std::endl;
    std::cout << " Average GFLOPS " << gflop / (sumTime / timeNumber)<<std::endl;

// ANTY-DCE: użyj C po pomiarze
    volatile double checksum = 0;
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            checksum += C[i][j];
    std::cout << " checksum=" << checksum << "\n";

    return 0;
}
