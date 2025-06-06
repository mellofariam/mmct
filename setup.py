from setuptools import setup, find_packages

setup(
    name="mmct",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "biopython",
        "PyCifRW",
        "openmm",
        "pdbfixer",
        "pandas",
    ],
    author="Matheus F. Mello",
    author_email="matheus@irce.edu",
    description="A package with tools for Modeling Molecular Complexes",
    url="https://github.com/mellofariam/mmct",
    license="GPL-2.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
    ],
    python_requires=">=3.6",
)
