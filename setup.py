"""Setup script for tamasha package."""

from setuptools import find_packages, setup

setup(
    name="tamasha",
    version="0.1.0",
    description="Bollywood movie intelligence platform",
    author="Tamasha Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
)
