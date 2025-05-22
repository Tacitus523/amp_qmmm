from setuptools import setup, find_packages

setup(
    name="amp_qmmm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "ase",
        "torchmetrics",
        "torchlayers",
        "pyyaml",
    ],
)