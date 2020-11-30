from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as f:
    readme = f.read()

with open("VERSION", "r") as f:
    version = f.read().strip()


install_requires = ["dask", "distributed", "pillow", "torch"]
testing_deps = ["black", "flake8", "mypy", "pytest", "pytest-cov"]

setup(
    name="dask-pytorch-ddp",
    version=version,
    maintainer="Saturn Cloud Developers",
    maintainer_email="open-source@saturncloud.io",
    license="BSD 3-clause",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="saturn cloud dask pytorch torch",
    description="library for setting up torch DDP on a dask cluster",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/saturncloud/dask-pytorch-ddp",
    project_urls={
        "Documentation": "https://github.com/saturncloud/dask-pytorch-ddp",
        "Source": "https://github.com/saturncloud/dask-pytorch-ddp",
        "Issue Tracker": "https://github.com/saturncloud/dask-pytorch-ddp/issues",
    },
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    extras_require={"dev": install_requires + testing_deps},
    test_suite="tests",
    tests_require=testing_deps,
    zip_safe=False,
)
