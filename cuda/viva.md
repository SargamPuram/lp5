# CUDA Vector Addition & Matrix Multiplication — Complete Viva Guide
### LP5 | Group A | Assignment 4A & 4B

---

> **How to use:** Read question, cover answer, say it out loud.
> If you can say it confidently without reading — you're ready.

---

## SECTION 1 — CUDA Fundamentals

---

### Q. What is CUDA?

CUDA stands for Compute Unified Device Architecture. It is a parallel computing platform and programming model developed by NVIDIA. It allows developers to use NVIDIA GPUs to accelerate computationally intensive tasks by running thousands of threads simultaneously. CUDA supports C, C++, and Python and provides APIs like cudaMalloc, cudaMemcpy, and cudaFree to manage GPU memory. It is widely used in scientific computing, machine learning, image processing, and deep learning.

---

### Q. What is the difference between CPU and GPU?

| | CPU | GPU |
|---|---|---|
| Cores | Few powerful cores (4–32) | Thousands of simple cores (thousands+) |
| Design goal | Low latency for sequential tasks | High throughput for parallel tasks |
| Best for | Complex logic, branching, OS tasks | Simple repetitive math on large data |
| Memory | Large RAM (GBs) | Dedicated VRAM (GBs, high bandwidth) |
| Thread count | Tens of threads efficiently | Millions of threads simultaneously |
| Example | Intel Core i9 | NVIDIA T4, A100, RTX 4090 |

---

### Q. What is a CUDA Kernel?

A kernel is a function that runs on the GPU, executed by many threads in parallel. It is declared using the `__global__` keyword. It is called from CPU code using the `<<<blocks, threads>>>` syntax. Each thread executes the same kernel code but on different data — this is called SIMT (Single Instruction Multiple Threads).

```cuda
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}
```

---

### Q. What are threads, blocks, and grids in CUDA?

- **Thread** — the smallest unit of execution. One thread processes one element (or one row/column).
- **Block** — a group of threads that run together and can share memory. Max 1024 threads per block.
- **Grid** — a collection of blocks. The entire kernel launch is one grid.

```
Grid
├── Block (0,0)  ── Thread(0,0), Thread(0,1), Thread(1,0)...
├── Block (0,1)  ── Thread(0,0), Thread(0,1)...
├── Block (1,0)  ── ...
```

---

### Q. What is threadIdx, blockIdx, blockDim?

These are built-in CUDA variables available inside every kernel:

| Variable | Meaning |
|---|---|
| `threadIdx.x` | Index of this thread within its block (x dimension) |
| `blockIdx.x` | Index of this block within the grid |
| `blockDim.x` | Total number of threads per block |
| `gridDim.x` | Total number of blocks in the grid |

**Global thread index formula:**
```cuda
int i = blockIdx.x * blockDim.x + threadIdx.x;
```
This gives each thread a unique index across the entire grid.

---

### Q. What is the CUDA memory model?

| Memory Type | Location | Speed | Scope |
|---|---|---|---|
| Global memory | GPU DRAM | Slow | All threads, entire grid |
| Shared memory | On-chip (per block) | Very fast | Threads within same block |
| Local memory | GPU DRAM | Slow | Single thread only |
| Register | On-chip (per thread) | Fastest | Single thread only |
| Host memory | CPU RAM | N/A — needs copy | CPU only |

In our programs we use global memory — cudaMalloc allocates on GPU global memory.

---

### Q. What is cudaMalloc, cudaMemcpy, cudaFree?

```cuda
// Allocate memory on GPU
cudaMalloc((void **)&d_A, size);

// Copy from CPU to GPU
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

// Copy from GPU back to CPU
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_A);
```

- `cudaMalloc` — like malloc but allocates on GPU global memory
- `cudaMemcpy` — transfers data between CPU and GPU (or GPU to GPU)
- `cudaFree` — like free but for GPU memory
- Host = CPU, Device = GPU — convention used throughout CUDA

---

### Q. What are CUDA Events and why do we use them for timing?

CUDA kernels execute asynchronously — the CPU launches the kernel and immediately continues without waiting. `chrono` on the CPU cannot accurately time GPU execution because the CPU timer stops before the GPU finishes.

CUDA Events are GPU-side timestamps:

```cuda
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);          // GPU records timestamp here
kernel<<<blocks, threads>>>();   // kernel executes
cudaEventRecord(stop);           // GPU records timestamp here

cudaEventSynchronize(stop);      // CPU waits for GPU to reach this point
float ms = 0;
cudaEventElapsedTime(&ms, start, stop);  // GPU-side elapsed time
```

This gives accurate GPU-only execution time excluding memory transfer overhead.

---

### Q. What is the compile command for CUDA?

```bash
nvcc filename.cu -o outputname
./outputname
```

`nvcc` is NVIDIA's CUDA compiler. It compiles both the CPU portions (using g++ internally) and GPU kernel code (for the target GPU architecture). The `.cu` extension tells nvcc this file contains CUDA code.

---

## SECTION 2 — Vector Addition

---

### Q. What is vector addition and why is it good for GPU?

Vector addition adds corresponding elements of two arrays: `C[i] = A[i] + B[i]`. Each element is completely independent — computing C[0] has no effect on C[1]. This is embarrassingly parallel — perfect for GPU because each thread can handle one element with zero communication between threads.

---

### Q. Explain the vector addition kernel line by line

```cuda
__global__ void vectorAdd(float *A, float *B, float *C, int n) {
```
> *"`__global__` means this runs on GPU but is called from CPU. Takes three GPU pointers and size n."*

```cuda
    int i = blockIdx.x * blockDim.x + threadIdx.x;
```
> *"Computes this thread's unique global index. blockIdx.x is which block, blockDim.x is threads per block, threadIdx.x is position within the block. Together they give a unique i for every thread."*

```cuda
    if (i < n) {
        C[i] = A[i] + B[i];
    }
```
> *"Bounds check — last block may have more threads than remaining elements. Only threads with valid index do work. Each thread adds one pair of elements."*

---

### Q. How are blocks and threads calculated for vector addition?

```cuda
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
```

For n = 1,000,000:
- threadsPerBlock = 256
- blocksPerGrid = (1,000,000 + 255) / 256 = 3,907 blocks
- Total threads = 3,907 × 256 = 1,000,192 — slightly more than n
- The `if (i < n)` guard handles the extra threads safely

The formula `(n + threadsPerBlock - 1) / threadsPerBlock` is ceiling division — ensures we have enough blocks to cover all n elements.

---

### Q. What are the steps of vector addition in CUDA?

1. **Define size** — n = 1,000,000, size = n × sizeof(float)
2. **Allocate host memory** — malloc for h_A, h_B, h_C
3. **Initialize vectors** — fill h_A = 1.0, h_B = 2.0 on CPU
4. **Allocate device memory** — cudaMalloc for d_A, d_B, d_C
5. **Copy host to device** — cudaMemcpy HostToDevice for A and B
6. **Launch kernel** — `vectorAdd<<<blocksPerGrid, threadsPerBlock>>>`
7. **Synchronize** — cudaEventSynchronize waits for GPU to finish
8. **Copy device to host** — cudaMemcpy DeviceToHost for C
9. **Verify** — print first 5 results (1+2=3 each)
10. **Free memory** — cudaFree for GPU, free for CPU

---

### Q. What output did your vector addition program give?

```
--- First 5 Results ---
1 + 2 = 3
1 + 2 = 3
1 + 2 = 3
1 + 2 = 3
1 + 2 = 3
...
[Success] Processed 1000000 elements.
[TIME] GPU Execution Time: 99.9829 ms
```

All 1 million elements correctly computed as 1.0 + 2.0 = 3.0.

---

## SECTION 3 — Matrix Multiplication

---

### Q. What is matrix multiplication?

For two N×N matrices A and B, the result C is computed as:
```
C[row][col] = sum of A[row][k] * B[k][col] for k = 0 to N-1
```

Each element of C requires N multiplications and N-1 additions. For an N×N matrix there are N² output elements. Sequential time complexity is O(N³). This is compute-intensive and perfect for GPU parallelization.

---

### Q. How is matrix multiplication parallelized in CUDA?

Each thread computes exactly one element of the output matrix C. With N² threads running simultaneously, all elements of C are computed in parallel. The key formula to map thread to matrix position:

```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
```

2D thread indexing maps naturally to 2D matrix structure.

---

### Q. Explain the matrix multiplication kernel line by line

```cuda
__global__ void matrixMultiply(int *A, int *B, int *C, int N) {
```
> *"GPU kernel taking flattened 1D pointers for the 2D matrices and matrix size N."*

```cuda
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
```
> *"Each thread computes its row and column in the output matrix using 2D indexing. blockIdx.y and threadIdx.y handle rows, blockIdx.x and threadIdx.x handle columns."*

```cuda
    if (row < N && col < N) {
```
> *"Bounds check for both dimensions — last row of blocks and last column of blocks may have extra threads beyond the matrix boundary."*

```cuda
        int sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
```
> *"This thread computes the dot product of row `row` of A with column `col` of B. A[row*N+k] accesses row-major flattened 2D array. This loop runs N times — each thread does N multiply-adds."*

```cuda
        C[row * N + col] = sum;
    }
}
```
> *"Store the computed element into the correct position in C."*

---

### Q. What is TILE_SIZE and why is it 16?

```cuda
#define TILE_SIZE 16
dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);  // 16×16 = 256 threads per block
```

TILE_SIZE = 16 gives 16×16 = 256 threads per block. This is a standard choice because:
- 256 is a multiple of the GPU warp size (32) — good hardware utilization
- 256 threads per block is well within the 1024 thread limit
- 16×16 tiles map naturally to matrix subregions
- It balances occupancy (blocks resident on GPU) with register usage

---

### Q. How are 2D blocks and grids calculated for matrix multiplication?

```cuda
dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);  // 16×16 block
dim3 blocksPerGrid(
    (N + TILE_SIZE - 1) / TILE_SIZE,   // columns
    (N + TILE_SIZE - 1) / TILE_SIZE    // rows
);
```

For N = 987:
- blocksPerGrid = ceil(987/16) × ceil(987/16) = 62 × 62 = 3844 blocks
- Total threads = 3844 × 256 = 983,864 threads
- All computing simultaneously — one per output element

---

### Q. How are 2D matrices stored in 1D memory in CUDA?

Matrices are stored in row-major order — rows laid out consecutively in memory:

```
Matrix A[3][3] stored as: A[0][0], A[0][1], A[0][2], A[1][0], A[1][1]...

Access formula: A[row][col] → A[row * N + col]
```

This is why the kernel uses `A[row * N + k]` and `B[k * N + col]` — converting 2D logical indices to 1D physical memory addresses.

---

### Q. What output did your matrix multiplication give?

For 2×2 manual input:
```
A = [[1,2],[3,4]]    B = [[5,6],[7,8]]

Result C:
[1×5 + 2×7, 1×6 + 2×8]   =  [19, 22]
[3×5 + 4×7, 3×6 + 4×8]   =  [43, 50]

[TIME] GPU Execution Time: 0.202752 ms
```

For 987×987 auto-generated:
```
[TIME] GPU Execution Time: 7.22298 ms
```
987×987 matrix multiplication — nearly 1 billion multiply-add operations — in 7 milliseconds on GPU.

---

## SECTION 4 — Official Viva Questions

---

### Q. What are the advantages of using CUDA for matrix multiplication compared to CPU?

*"Several major advantages:*

*First — massive parallelism. GPU has thousands of cores. For an N×N matrix, all N² output elements are computed simultaneously. CPU would compute them sequentially.*

*Second — speed. Our 987×987 matrix multiplication completed in 7.2ms on GPU. Sequential CPU implementation would take seconds.*

*Third — high memory bandwidth. GPU has dedicated VRAM with much higher bandwidth than CPU RAM — essential for memory-bound operations like matrix access.*

*Fourth — scalability. As matrix size grows, we simply add more blocks — the GPU naturally scales.*

*The tradeoff is memory transfer overhead — copying data between CPU and GPU RAM takes time. For very small matrices CPU may actually be faster due to this overhead."*

---

### Q. How do you handle matrices too large to fit in GPU memory?

*"Several strategies:*

*First — tiling or blocking. Divide the matrix into smaller tiles that fit in GPU memory. Process one tile at a time, keeping partial results on GPU.*

*Second — streaming. Use CUDA streams to overlap data transfer with computation. While the GPU computes on one tile, the CPU transfers the next tile.*

*Third — out-of-core computation. Keep the full matrix on CPU RAM and transfer only the portions needed for current computation.*

*Fourth — use shared memory tiling in the kernel. Load a TILE_SIZE × TILE_SIZE submatrix into fast shared memory, compute on it, then load the next tile. This reduces global memory accesses significantly.*

*For extremely large matrices, distributed approaches using multiple GPUs across multiple machines are used."*

---

### Q. How do you optimize the CUDA matrix multiplication program?

*"Several optimization techniques:*

*First — shared memory tiling. Instead of each thread reading from slow global memory N times, load a tile into shared memory once and have all threads in the block reuse it. This reduces global memory reads by TILE_SIZE times.*

*Second — coalesced memory access. Ensure threads in a warp access consecutive memory addresses. Our column-major access for B (`B[k*N+col]`) is not coalesced — matrix B transposition or shared memory can fix this.*

*Third — larger tile sizes. Bigger tiles mean more data reuse in shared memory but require more shared memory per block.*

*Fourth — use tensor cores. Modern NVIDIA GPUs have specialized tensor core hardware for matrix operations — accessible via cuBLAS library.*

*Fifth — loop unrolling. The inner k loop can be unrolled to reduce loop overhead.*

*Our current implementation is the naive version — correct and demonstrably fast but not fully optimized."*

---

### Q. How do you ensure correctness of CUDA matrix multiplication?

*"Three approaches we use:*

*First — manual verification for small matrices. For N≤5 we print all three matrices and manually verify. For the 2×2 test: A=[[1,2],[3,4]], B=[[5,6],[7,8]], expected C=[[19,22],[43,50]] — our output matches exactly.*

*Second — comparison with CPU result. Run the same multiplication sequentially on CPU and compare element by element with GPU output. Any mismatch indicates a kernel bug.*

*Third — known input verification. Use inputs where the expected answer is known — like identity matrix multiplication, or all-ones matrices where C[i][j] = N for all elements.*

*In production, cudaGetLastError() after kernel launch and after cudaMemcpy catches runtime errors."*

---

### Q. How do you add two large vectors using CUDA? Describe the steps.

1. Define n = 1,000,000 and allocate host memory with malloc
2. Initialize h_A = 1.0 and h_B = 2.0 for all elements on CPU
3. Allocate device memory with cudaMalloc for d_A, d_B, d_C
4. Copy h_A and h_B to GPU with cudaMemcpy HostToDevice
5. Calculate threadsPerBlock = 256, blocksPerGrid = ceil(n/256) = 3907
6. Launch kernel `vectorAdd<<<3907, 256>>>` — 1 million threads run simultaneously
7. Each thread computes C[i] = A[i] + B[i] for its unique index i
8. Record GPU time using CUDA Events
9. Copy d_C back to h_C with cudaMemcpy DeviceToHost
10. Verify first 5 results, print timing, free all memory

---

## SECTION 5 — Tricky Examiner Questions

---

### Q. Why do we use `if (i < n)` in the vector addition kernel?

The number of threads launched is always a multiple of threadsPerBlock (256). For n = 1,000,000 we launch 3,907 × 256 = 1,000,192 threads — 192 more than needed. Without the bounds check, these 192 extra threads would access memory beyond the allocated array — undefined behavior and potential crash. The `if (i < n)` guard ensures only valid threads write to C.

---

### Q. Why is the matrix stored as 1D array instead of 2D array in CUDA?

CUDA cudaMalloc allocates a flat 1D block of memory. A 2D array in C++ (int A[N][N]) requires pointer-of-pointers which cannot be directly passed to the GPU — each inner pointer would be a CPU address invalid on GPU. Flattening to 1D with the formula `A[row * N + col]` gives a single contiguous block that cudaMalloc and cudaMemcpy can handle directly and efficiently.

---

### Q. What is a warp in CUDA?

A warp is a group of 32 threads that execute the same instruction simultaneously on the GPU hardware. It is the fundamental scheduling unit — the GPU hardware always executes threads in warps of 32. If threads within a warp take different branches (if-else), the GPU executes both branches serially for the different threads — called warp divergence. Choosing threadsPerBlock as a multiple of 32 (like 256) ensures no warp is partially empty.

---

### Q. What is the difference between `__global__`, `__device__`, and `__host__`?

| Qualifier | Runs on | Called from | Use |
|---|---|---|---|
| `__global__` | GPU | CPU | Kernel functions — entry point for GPU execution |
| `__device__` | GPU | GPU only | Helper functions called from within a kernel |
| `__host__` | CPU | CPU | Regular CPU functions (default, optional to write) |
| `__host__ __device__` | Both | Both | Utility functions usable on both CPU and GPU |

---

### Q. What is cudaEventSynchronize and why is it needed?

CUDA kernel launches are asynchronous — `kernel<<<...>>>()` returns immediately to the CPU without waiting for the GPU to finish. If we call `cudaEventElapsedTime` immediately, the stop event may not have been recorded yet. `cudaEventSynchronize(stop)` blocks the CPU until the GPU has recorded the stop event — guaranteeing the timing is complete and the result is ready to copy back.

---

### Q. Why does vector addition take ~100ms but matrix multiplication of 987×987 takes only 7ms?

This seems counterintuitive but makes sense:

Vector addition on 1M elements is **memory-bound** — threads do almost no computation (just one addition each) but must read two values and write one from/to slow global memory. Memory bandwidth is the bottleneck.

Matrix multiplication on 987×987 is **compute-bound** — each thread does N=987 multiply-adds. The GPU's thousands of arithmetic units are fully utilized. Compute throughput dominates over memory access.

Also the 100ms includes first-time GPU initialization overhead — subsequent runs on the same GPU would be faster.

---

### Q. What is the h_ and d_ naming convention?

It is a CUDA programming convention:
- `h_` prefix = Host variable (lives in CPU RAM)
- `d_` prefix = Device variable (lives in GPU VRAM)

```cuda
float *h_A;        // CPU memory — used for input/output by CPU
float *d_A;        // GPU memory — used by kernel during computation
cudaMalloc(&d_A);  // allocate on GPU
cudaMemcpy(d_A, h_A, ...);  // copy CPU → GPU
```

This makes it immediately clear which memory space a pointer belongs to, preventing accidental use of CPU pointers in GPU kernels and vice versa.

---

### Q. What happens if you forget cudaFree or free?

- Forgetting `cudaFree` — GPU memory leak. GPU VRAM is not released until the program exits. For long-running programs or loops, this depletes GPU memory and eventually causes cudaMalloc to fail.
- Forgetting `free` — CPU memory leak. Same issue on the CPU side.
- In our programs both are always called at the end — clean memory management.

---

### Q. Can CUDA work without an NVIDIA GPU?

No. CUDA is NVIDIA proprietary technology and only works on NVIDIA GPUs. Alternatives for other hardware:
- **OpenCL** — open standard, works on AMD, Intel, NVIDIA GPUs
- **HIP** — AMD's CUDA-like API for AMD GPUs
- **SYCL** — modern cross-platform GPU programming
- **Metal** — Apple GPU programming

For our practical, Google Colab with T4 GPU provides the CUDA environment for free.

---

## SECTION 6 — Quick Revision Flashcards

| Question | One line answer |
|---|---|
| What is CUDA? | NVIDIA's parallel computing platform for GPU programming |
| What is a kernel? | GPU function declared with `__global__`, run by thousands of threads |
| What is a thread? | Smallest execution unit — processes one element |
| What is a block? | Group of threads that share memory and run together |
| What is a grid? | Collection of all blocks for one kernel launch |
| What is threadIdx? | Thread's index within its block |
| What is blockIdx? | Block's index within the grid |
| Global index formula? | `blockIdx.x * blockDim.x + threadIdx.x` |
| What is cudaMalloc? | Allocate memory on GPU |
| What is cudaMemcpy? | Transfer data between CPU and GPU |
| What is cudaFree? | Free GPU memory |
| Why CUDA Events for timing? | Kernels are async — CPU timer misses GPU execution time |
| What is cudaEventSynchronize? | Block CPU until GPU reaches that event |
| Compile command? | `nvcc filename.cu -o outputname` |
| Why `if (i < n)` guard? | Last block has extra threads beyond array size |
| Why 1D array for matrix? | cudaMalloc needs contiguous flat memory — no pointer-of-pointers |
| Access formula for 2D matrix? | `A[row * N + col]` |
| What is TILE_SIZE 16? | 16×16 = 256 threads per block — multiple of warp size 32 |
| What is h_ prefix? | Host (CPU) memory |
| What is d_ prefix? | Device (GPU) memory |
| What is a warp? | 32 threads executing same instruction simultaneously |
| Why vector add slower than matrix mul? | Memory-bound vs compute-bound — GPU arithmetic units idle for vector add |
| What is `__global__`? | Runs on GPU, called from CPU |
| What is `__device__`? | Runs on GPU, called from GPU only |
| What if no NVIDIA GPU? | Use OpenCL, HIP, or Google Colab's free T4 GPU |

---

*CUDA is the most impressive practical — you're using a real GPU with thousands of cores.*
*Key message: one thread per element, global index formula, host-device memory transfers.*
*Know the full flow: malloc → cudaMalloc → cudaMemcpy → kernel → sync → cudaMemcpy back → free.*