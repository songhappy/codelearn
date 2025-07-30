#include <sycl/sycl.hpp>
#include <chrono>
#include <iostream>

using namespace sycl;

constexpr size_t GB = 1024 * 1024 * 1024;
constexpr size_t N = 16L * GB / sizeof(float);  // 16 GB total
constexpr int VEC_SIZE = 8;                   // Use vec<float, 8>
constexpr int LOOP_ITERS = 1000;

int main() {
    queue q0{gpu_selector_v};
    queue q1{gpu_selector_v};
    queue q2{gpu_selector_v};
    queue q3{gpu_selector_v};

    std::cout << "Running on device: " << q0.get_device().get_info<info::device::name>() << "\n";

    // Allocate and split memory into 4 slices
    size_t N_chunk = N / 4;
    float* a_dev = malloc_device<float>(N, q0);
    float* b_dev = malloc_device<float>(N, q0);
    float* c_dev = malloc_device<float>(N, q0);
    float scalar = 3.0f;

    // Warm-up triad kernel
    q0.parallel_for(nd_range<1>{range<1>{N_chunk / VEC_SIZE}, range<1>{1024}}, [=](nd_item<1> it) {
        size_t i = it.get_global_id(0);
        ((vec<float, VEC_SIZE>*)a_dev)[i] = ((vec<float, VEC_SIZE>*)b_dev)[i] + scalar * ((vec<float, VEC_SIZE>*)c_dev)[i];
    });
    q1.parallel_for(nd_range<1>{range<1>{N_chunk / VEC_SIZE}, range<1>{1024}}, [=](nd_item<1> it) {
        size_t i = it.get_global_id(0);
        ((vec<float, VEC_SIZE>*)(a_dev + N_chunk))[i] = ((vec<float, VEC_SIZE>*)(b_dev + N_chunk))[i] + scalar * ((vec<float, VEC_SIZE>*)(c_dev + N_chunk))[i];
    });
    q2.parallel_for(nd_range<1>{range<1>{N_chunk / VEC_SIZE}, range<1>{1024}}, [=](nd_item<1> it) {
        size_t i = it.get_global_id(0);
        ((vec<float, VEC_SIZE>*)(a_dev + 2 * N_chunk))[i] = ((vec<float, VEC_SIZE>*)(b_dev + 2 * N_chunk))[i] + scalar * ((vec<float, VEC_SIZE>*)(c_dev + 2 * N_chunk))[i];
    });
    q3.parallel_for(nd_range<1>{range<1>{N_chunk / VEC_SIZE}, range<1>{1024}}, [=](nd_item<1> it) {
        size_t i = it.get_global_id(0);
        ((vec<float, VEC_SIZE>*)(a_dev + 3 * N_chunk))[i] = ((vec<float, VEC_SIZE>*)(b_dev + 3 * N_chunk))[i] + scalar * ((vec<float, VEC_SIZE>*)(c_dev + 3 * N_chunk))[i];
    });
    q0.wait(); q1.wait(); q2.wait(); q3.wait();

    // Timed triad loop across all queues
    auto start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < LOOP_ITERS; ++k) {
        q0.parallel_for(nd_range<1>{range<1>{N_chunk / VEC_SIZE}, range<1>{1024}}, [=](nd_item<1> it) {
            size_t i = it.get_global_id(0);
            ((vec<float, VEC_SIZE>*)a_dev)[i] = ((vec<float, VEC_SIZE>*)b_dev)[i] + scalar * ((vec<float, VEC_SIZE>*)c_dev)[i];
        });
        q1.parallel_for(nd_range<1>{range<1>{N_chunk / VEC_SIZE}, range<1>{1024}}, [=](nd_item<1> it) {
            size_t i = it.get_global_id(0);
            ((vec<float, VEC_SIZE>*)(a_dev + N_chunk))[i] = ((vec<float, VEC_SIZE>*)(b_dev + N_chunk))[i] + scalar * ((vec<float, VEC_SIZE>*)(c_dev + N_chunk))[i];
        });
        q2.parallel_for(nd_range<1>{range<1>{N_chunk / VEC_SIZE}, range<1>{1024}}, [=](nd_item<1> it) {
            size_t i = it.get_global_id(0);
            ((vec<float, VEC_SIZE>*)(a_dev + 2 * N_chunk))[i] = ((vec<float, VEC_SIZE>*)(b_dev + 2 * N_chunk))[i] + scalar * ((vec<float, VEC_SIZE>*)(c_dev + 2 * N_chunk))[i];
        });
        q3.parallel_for(nd_range<1>{range<1>{N_chunk / VEC_SIZE}, range<1>{1024}}, [=](nd_item<1> it) {
            size_t i = it.get_global_id(0);
            ((vec<float, VEC_SIZE>*)(a_dev + 3 * N_chunk))[i] = ((vec<float, VEC_SIZE>*)(b_dev + 3 * N_chunk))[i] + scalar * ((vec<float, VEC_SIZE>*)(c_dev + 3 * N_chunk))[i];
        });
    }
    q0.wait(); q1.wait(); q2.wait(); q3.wait();

    double seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();
    double total_bytes = static_cast<double>(N) * sizeof(float) * 3 * LOOP_ITERS;
    double gbps = total_bytes / seconds / 1e9;

    std::cout << "Sustained XPU bandwidth (triad, multi-queue, vec8): " << gbps << " GB/s\n";

    free(a_dev, q0); free(b_dev, q0); free(c_dev, q0);
}
