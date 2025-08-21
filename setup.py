#!/usr/bin/env python
# coding: utf-8

from setuptools import find_packages, setup

setup(
    name="genome_diffusion",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "torch>=1.9.0",
        "pytorch-lightning>=1.5.0",
        "pyyaml>=6.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "test": [
            "pytest>=6.0.0",
            "pytest-cov>=2.8.0",
        ],
        "dev": [
            "black>=21.7b0",
            "isort>=5.0.0",
            "mypy>=0.910",
            "flake8>=3.9.0",
        ],
    },
)
