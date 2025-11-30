from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="AntiCAP",
    version="3.3.3",
    author="NewArk81",
    description="AntiCAP",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/81NewArk/AntiCAP",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'onnxruntime',
        'Pillow',
        'opencv-python',
        'ultralytics',
        'requests',
        'tqdm',
    ],
    python_requires='<=3.13',
    include_package_data=True,
)