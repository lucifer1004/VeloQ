// Synthetic CUDA kernels used to populate the source_metric_basic
// fixture for the veloq `ncu source-metrics` smoke tests. Per
// [[no real names in test fixtures]] the kernel names + variable
// naming are deliberately generic — no internal product / model /
// workload identifiers.
//
// Two kernels:
//
//   launch:0 → `synthetic_bank_conflict_kernel`
//     One shared-memory load pattern with an intentional 32-way
//     bank conflict (every warp lane reads the same bank) and one
//     trivial arithmetic op. Exercises both `Section.SourceMetrics`
//     and `ProfilerSourceMetricTable` body-item paths under
//     `--set full`, plus the additive `derived__memory_l1_conflicts_*`
//     family.
//
//   launch:1 → `synthetic_long_stall_kernel`
//     Pointer-chase loop with dependent global loads. Each
//     iteration's address depends on the previous load's value, so
//     there's no ILP and every load can stall on `long_scoreboard`.
//     Sized large enough (256 blocks × 256 threads × 256 iters) for
//     PC sampling to accumulate per-PC instances in the
//     `warpsampling:smsp__pcsamp_warps_issue_stalled_*` family.

#include <cuda_runtime.h>
#include <cstdio>

constexpr int BLOCK = 256;

// Bank-conflict kernel ------------------------------------------------------

__shared__ int bank_smem[BLOCK];

__global__ void synthetic_bank_conflict_kernel(int* out) {
    int tid = threadIdx.x;

    // Stripe writes — no conflict on store.
    bank_smem[tid] = tid;
    __syncthreads();

    // 32-way bank conflict read: every lane reads from index 0,
    // which maps to bank 0. NCU's SourceCounters section reports
    // the conflict count on this exact source line.
    int v = bank_smem[0];

    // Arithmetic op on a different source line so the kernel has
    // a second attributable line in the line-table.
    v = v + tid * 2;

    if (tid == 0) {
        out[blockIdx.x] = v;
    }
}

// Long-stall pointer-chase kernel -------------------------------------------

constexpr int CHAIN_LOG2 = 18;             // chain[] = 1 MiB of ints
constexpr int CHAIN_N    = 1 << CHAIN_LOG2;
constexpr int CHAIN_MASK = CHAIN_N - 1;
constexpr int STALL_GRID = 256;
constexpr int STALL_ITERS = 256;

__global__ void synthetic_long_stall_kernel(const int* __restrict__ chain,
                                            int* __restrict__ out,
                                            int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x = idx & CHAIN_MASK;

    // Pointer-chase: each iteration's load address depends on the
    // previous load's value. No ILP, so each load can stall on
    // `long_scoreboard` waiting for the cache fill. PC sampling
    // accumulates per-PC stall instances on this loop body.
    for (int i = 0; i < iters; ++i) {
        x = chain[x & CHAIN_MASK];
    }

    if (idx < (STALL_GRID * BLOCK)) {
        out[idx] = x;
    }
}

// Host driver ---------------------------------------------------------------

int main() {
    // Bank-conflict launch (small, one block — same as before).
    int* d_bank_out = nullptr;
    cudaMalloc(&d_bank_out, sizeof(int) * 64);
    synthetic_bank_conflict_kernel<<<1, BLOCK>>>(d_bank_out);
    cudaDeviceSynchronize();
    cudaFree(d_bank_out);

    // Long-stall launch. Allocate + init the pointer-chase chain on
    // the host (a simple LCG permutation so the values are stable
    // across machines), copy to device, run.
    int* h_chain = static_cast<int*>(malloc(sizeof(int) * CHAIN_N));
    unsigned int s = 1u;
    for (int i = 0; i < CHAIN_N; ++i) {
        s = s * 1664525u + 1013904223u;
        h_chain[i] = static_cast<int>(s & CHAIN_MASK);
    }
    int* d_chain = nullptr;
    cudaMalloc(&d_chain, sizeof(int) * CHAIN_N);
    cudaMemcpy(d_chain, h_chain, sizeof(int) * CHAIN_N, cudaMemcpyHostToDevice);
    free(h_chain);

    int* d_stall_out = nullptr;
    cudaMalloc(&d_stall_out, sizeof(int) * STALL_GRID * BLOCK);
    synthetic_long_stall_kernel<<<STALL_GRID, BLOCK>>>(d_chain, d_stall_out, STALL_ITERS);
    cudaDeviceSynchronize();

    cudaFree(d_stall_out);
    cudaFree(d_chain);
    return 0;
}
