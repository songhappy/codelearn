
References:
https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html#:~:text=The%20GPU%20is%20a%20highly,%2C%20and%20high%2Dbandwidth%20DRAM.
https://zhuanlan.zhihu.com/p/678001378
https://siboehm.com/articles/22/CUDA-MMM


## GPU Architecture Fundamentals
[GPU Architecture Diagram simplified](images/GPU_architecture_NV.png) 

[GPU Architecture Diagram](images/GPU_architecture.png)
The GPU is a highly parallel processor architecture, composed of processing elements and a memory hierarchy. At a high level, NVIDIA® GPUs consist of a number of Streaming Multiprocessors (SMs), on-chip L2 cache, and high-bandwidth DRAM. Arithmetic and other instructions are executed by the SMs; data and code are accessed from DRAM via the L2 cache. 
As an example, an NVIDIA A100 GPU contains 108 SMs, a 40 MB L2 cache, and up to 2039 GB/s bandwidth (maximum rate at thich data ca be read from or written to the memory by a processor or GPU) from 80 GB of HBM2 (High Bandwith Memory) memory. DRAW(Dynamic Random Access Memory).

The A100 GPU has a total of 108 SMs.
Each SM contains:
64 CUDA cores for general-purpose arithmetic operations.
4 Tensor Cores for specialized matrix operations commonly used in AI and deep learning.
To summarize:
CUDA cores per SM: 64
Total SMs in A100: 108
Total CUDA cores in A100: 64×108=6,912
Total tensor cores in A100: 4x108=432
FP 32 flops: 19.5 TFLOPS
BF16 Tensor Core: 312 TFLOPS

+----------------------------------+
|       Streaming Multiprocessor   |
|                                  |
| +---------+   +----------------+ |
| | CUDA    |   | Tensor Cores    | |
| | Cores   |   | (Matrix Ops)    | |
| | (General|   |                 | |
| | Purpose)|   +----------------+ |
| +---------+                      |
|   |                              |
| Registers | Shared Memory | Cache |
+----------------------------------+

## GPU Excution Model
### definitions and terminolodgies 
1. Threads
Definition: The smallest unit of execution in CUDA. Each thread runs a kernel and performs computations.
CUDA Cores: Handle one thread at a time, executing instructions from the assigned thread in a warp.
Tensor Cores: Can perform operations for multiple threads simultaneously in matrix operations, significantly increasing throughput for workloads such as deep learning.
Execution: Threads can be organized into thread blocks which are organized into warps(32 threads) for execution.
2. Thread Blocks
Definition: A group of threads that execute a kernel. A thread block can contain up to 1024 threads.
Characteristics: Threads within the same block can communicate with each other and share data through shared memory.
Execution Model: When a kernel is launched, it is executed by multiple thread blocks concurrently.
3. Warps
Definition: A warp is a group of 32 threads that execute the same instruction simultaneously (SIMD - Single Instruction, Multiple Data).
Characteristics: Warps allow for efficient scheduling and execution by the GPU. When threads in a warp diverge due to conditional statements, it can lead to underutilization of resources.
Mapping: Each thread block is divided into multiple warps, depending on the block size.
4. Streaming Multiprocessors (SMs)
Definition: A physical unit in the GPU that contains several CUDA cores and manages multiple warps. Each SM can handle multiple warps at a time, switching between them to hide memory latency.
Characteristics: Each SM can execute many threads simultaneously and features resources like registers, shared memory, and control logic for scheduling warps.
Execution: When a kernel is executed, thread blocks are assigned to SMs, which further schedule warps for execution.
5. Threads, thread blocks per SM [mapping](images/automatic-scalability-nv.png)
Each SM in the A100 can manage up to 2048 active threads concurrently. If your block size is 1024 threads, and the maximum threads per SM is 2048, then an SM can handle 2 blocks of 1024 threads each at a time. Each SM in an A100 GPU has the capacity to run up to 16 thread blocks simultaneously if each thread block is small.


### NVIDIA Parallelization Flow using example of A X B = C
1. Split the Matrix into Tiles
The matrices 𝐴 and 𝐵  are divided into smaller sub-matrices, or tiles. This is done to optimize memory access patterns and to fit the data into the limited shared memory available in the SM (Streaming Multiprocessor).

Example:
Divide 𝐴 into tiles of size 4×𝑛 (where 4 is the tile height, if A has 1024 rows then we will have 254 tiles of A).
Divide 𝐵 into tiles of size n×4 (where 4 is the tile width).
This division allows for localized data access, minimizing cache misses and improving performance.

2. Assign Tiles to Thread Blocks
Each tile is assigned to a thread block in the GPU. Each thread block can process a portion of the resulting matrix 𝐶.
Grid of Blocks: For a resulting tile 𝐶(𝑖,𝑗), you can assign a block of threads that will compute the contributions of a specific tile of 
A and B.
The number of thread blocks is determined by the dimensions of the output matrix C.

3. Use Shared Memory to Store Sub-Matrices
Once the tiles are assigned, they are loaded into shared memory, which is a faster memory space compared to global memory.
Loading Tiles: Each thread block loads a tile of A and a tile of B into shared memory.
This is done in a coalesced manner to maximize throughput.

4. Perform Parallel Computations Within Each Block
Each thread in the block computes a part of the final output matrix C by performing the matrix multiplication for its assigned elements.
Matrix Multiplication Logic:
For a thread at position (i,j), it calculates: C[i,j]+=A[i,k]∗B[k,j]
The calculation is performed for all k in the tile.
This allows each thread to work on different parts of the computation in parallel, leveraging the high level of concurrency provided by the GPU.

5. Accumulate Results to Form the Final Matrix
After all threads have completed their computations, the results from shared memory are written back to the global memory.

Final Assembly:
Each thread block writes its computed results back to the corresponding portion of matrix C in global memory.

### Optimize a cuda matmal kernel
https://siboehm.com/articles/22/CUDA-MMM



### Development steps using examples
