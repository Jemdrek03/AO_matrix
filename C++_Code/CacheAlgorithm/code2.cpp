#include <cstdint>
#include <iostream>
#include <ctime>
#include <cassert>
#include <atomic>

#define N 2048
static float A[N][N], B[N][N], C[N][N], BT[N][N];

uint64_t nanos(){
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (uint64_t)t.tv_sec*1000000000ull + (uint64_t)t.tv_nsec;
}

int main() {
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++) {
            A[i][j] = (i + j) * 0.001f;
            B[i][j] = ((i*37 + j*17) % 101) * 0.01f;
        }


    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            BT[j][i] = B[i][j];

    const int BY=32, BX=32, BK=32;

    std::atomic_thread_fence(std::memory_order_seq_cst);
    uint64_t start = nanos();

    for (int by=0; by<N; by+=BY)
        for (int bx=0; bx<N; bx+=BX)
            for (int y=by; y<by+BY; ++y)
                for (int x=bx; x<bx+BX; ++x) {
                    float acc = 0.0f;
                    for (int bk=0; bk<N; bk+=BK)
                        for (int k=bk; k<bk+BK; ++k)
                            acc += A[y][k] * BT[x][k];
                    C[y][x] = acc;
                }

    std::atomic_thread_fence(std::memory_order_seq_cst);
    uint64_t end = nanos();

    double seconds = (end-start) * 1e-9;
    double gflops  = (2.0 * N * N * (double)N) * 1e-9 / seconds;
    std::cout << "GFLOP/s " << gflops;

// ANTY-DCE: użyj C po pomiarze
    volatile double checksum = 0;
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            checksum += C[i][j];
    std::cout << " checksum=" << checksum << "\n";

    return 0;
}
