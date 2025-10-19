#!/usr/bin/env python3
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="booklet-reorder",
    version="1.1.1",
    author="Michael Greiner",
    description="Convert scanned saddle-stitched booklet PDFs into properly ordered pages for duplex printing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mgreiner/booklet-reorder",
    license="MIT",
    py_modules=["booklet_reorder"],
    python_requires=">=3.7",
    install_requires=[
        "PyPDF2>=3.0.0",
        "PyMuPDF>=1.23.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "booklet-reorder=booklet_reorder:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
        "Topic :: Printing",
    ],
)
