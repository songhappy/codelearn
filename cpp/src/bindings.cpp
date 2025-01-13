#include <pybind11/pybind11.h>

int add_numbers(int a, int b) {
    return a + b;
}

PYBIND11_MODULE(example, m) {
    m.doc() = "Example module"; // Optional docstring
    m.def("add_numbers", &add_numbers, "A function that adds two numbers");
}
