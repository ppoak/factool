import quool
from setuptools import setup, find_packages


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name = "factorlab",
    packages = ["factorlab"],
    author = "ppoak",
    author_email = "ppoak@foxmail.com",
    description = "Factor lab is a quant framework for factor research.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    keywords = ['quant', 'factor analysis', 'finance'],
    url = "https://github.com/ppoak/quool",
    version = quool.__version__,
    install_requires = [
        'numpy',
        'quool',
        'pandas',
        'joblib',
        'statsmodels',
    ],
)