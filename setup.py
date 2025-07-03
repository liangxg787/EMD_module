# -*- coding: UTF-8 -*-
"""
@Time : 03/07/2025 11:08
@Author : Xiaoguang Liang
@File : setup.py
@Project : EMD_module
"""
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

VERSION = "1.0.0"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="EMD_module",
    version=VERSION,
    author='Xiaoguang Liang',
    author_email='hplxg@hotmail.com',
    url='https://github.com/liangxg787/EMD_module.git',
    description="EMD_module: the Earth Mover's Distance (EMD) module",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license='MIT',
    keywords='EMD_module',
    include_package_data=True,
    zip_safe=False,
    packages=[],
    install_requires=[],
    entry_points={},
    ext_modules=[
        CUDAExtension('emd', [
            'emd.cpp',
            'emd_cuda.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: MIT License",
    ]
)
