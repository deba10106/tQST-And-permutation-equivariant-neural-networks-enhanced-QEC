from setuptools import setup, find_packages

setup(
    name="mlqec",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.5",
        "matplotlib>=3.4.3",
        "qiskit<1.0.0",
        "qiskit-aer<0.13.0",
        "qiskit-ibmq-provider<0.20.0",
        "qiskit-ignis<0.8.0",
        "tensorboard>=2.7.0"
    ]
)
