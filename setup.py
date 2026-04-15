from setuptools import setup, find_packages

setup(
    name="gpl-tokenizer",
    version="0.1.0",
    description="Geometric Primitive Language Tokenizer for SVG",
    author="Byun",
    author_email="igotthepower0128@gmail.com",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "ml": [
            "torch>=2.0",
            "transformers>=4.30",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
