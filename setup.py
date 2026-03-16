"""
Setup configuration for AutoML Hyperparameter Optimization.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="automl-hyperparameter-optimization",
    version="1.0.0",
    author="Gabriel Demetrios Lafis",
    description="AutoML system for hyperparameter optimization with grid, random, Bayesian and Hyperband search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/galafis/automl-hyperparameter-optimization",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    install_requires=[
        "optuna>=3.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.0.0",
            "matplotlib>=3.7.0",
            "plotly>=5.14.0",
        ],
    },
    keywords="automl, hyperparameter-optimization, bayesian-optimization, hyperband, grid-search",
)
