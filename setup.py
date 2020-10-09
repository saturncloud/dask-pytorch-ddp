from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as f:
    readme = f.read()

with open("VERSION", "r") as f:
    version = f.read().strip()

# prefect has to be 0.13.0 or newer to get Webhook storage
install_requires = ["cloudpickle", "dask-saturn>=0.0.4", "prefect>0.13.0", "requests"]
testing_deps = ["pytest", "pytest-cov", "responses"]

setup(
    name="prefect-saturn",
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
    keywords="saturn cloud prefect prefect",
    description="Client library for running Prefect Cloud flows in Saturn Cloud",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://saturncloud.io/",
    project_urls={
        "Documentation": "http://docs.saturncloud.io",
        "Source": "https://github.com/saturncloud/prefect-saturn",
        "Issue Tracker": "https://github.com/saturncloud/prefect-saturn/issues",
    },
    packages=find_packages(),
    install_requires=install_requires,
    python_requires=">=3.7",
    extras_require={"dev": install_requires + testing_deps},
    test_suite="tests",
    test_require=testing_deps,
    zip_safe=False,
)
