#include <torch/extension.h>

// Define the custom addition function
torch::Tensor custom_add(torch::Tensor a, torch::Tensor b) {
    // Ensure tensors are the same shape
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same size");
    return a + b; // Element-wise addition
}

// Define the Pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("custom_add", &custom_add, "A custom addition function");
}
