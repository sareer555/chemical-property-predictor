"""
Chemical Property Prediction - Setup Configuration
=====================================================

Production-quality setup for the cheminformatics ML pipeline.
Suitable for graduate-level research in computational chemistry.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="chemical-property-predictor",
    version="1.0.0",
    author="Computational Chemistry Research Group",
    author_email="research@example.edu",
    description="End-to-end ML pipeline for predicting chemical properties from molecular structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/username/chemical-property-predictor",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.7.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
        ],
    },
)
