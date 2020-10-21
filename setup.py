from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as f:
    readme = f.read()

with open("VERSION", "r") as f:
    version = f.read().strip()


install_requires = ["dask", "distributed", "torch"]
testing_deps = ["black", "pytest", "pytest-cov"]


setup(
    name="dask-pytorch",
    version=version,
    maintainer="Saturn Cloud Developers",
    maintainer_email="open-source@saturncloud.io",
    license="BSD 3-clause",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: System :: Distributed Computing",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    keywords="saturn cloud dask pytorch torch",
    description="library for setting up torch DDP on a dask cluster",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://saturncloud.io/",
    project_urls={
        "Documentation": "http://docs.saturncloud.io",
        "Source": "https://github.com/saturncloud/dask-pytorch",
        "Issue Tracker": "https://github.com/saturncloud/dask-pytorch/issues",
    },
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    extras_require={"dev": install_requires + testing_deps},
    test_suite="tests",
    test_require=testing_deps,
    zip_safe=False,
)
