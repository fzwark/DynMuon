import os
from setuptools import find_packages
from setuptools import setup

this_directory = os.path.dirname(__file__)
req_path = os.path.join(this_directory, "requirements_dion.txt")
req_dev_path = os.path.join(this_directory, "requirements_dev.txt")
req_train_path = os.path.join(this_directory, "requirements_train.txt")


def read_requirements(path):
    if not os.path.exists(path):
        print(f"Warning: requirements file {path} does not exist.")
        return []
    with open(path) as fp:
        return [
            line.strip() for line in fp if line.strip() and not line.startswith("#")
        ]


# requirements_dion contains the dependencies for the standalone optimizer
install_requires = read_requirements(req_path)

# requirements_dev contains the dependencies for development, e.g., testing, linting, etc.
install_dev_requires = install_requires + read_requirements(req_dev_path)

# requirements_train contains the dependencies for training, e.g., datasets, etc.
install_train_requires = install_requires + read_requirements(req_train_path)

setup(
    name="dynmuon",
    version="0.1.0",
    packages=find_packages(include=["dynmuon", "dynmuon.*"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    extras_require={
        "dev": install_dev_requires,
        "train": install_train_requires,
    },
)
