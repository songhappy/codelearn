import torch
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import pybind11

setup(
    name='custom_ops',
    version='0.1',
    ext_modules=[
        CppExtension(
            'custom_ops',
            ['cpp/src/custom_ops.cpp'],
            include_dirs=[
                pybind11.get_include(),
                *torch.utils.cpp_extension.include_paths(),
            ],
            libraries=['torch', 'torch_python'],
            language='c++',
        ),
    ],
    cmdclass={'build_ext': BuildExtension},
)
