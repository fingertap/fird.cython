from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

ext_modules = [
    Extension("fird.fird",
              sources=["fird/fird.pyx"],
              libraries=["m", "fird/fird.pxd"]  # Unix-like specific
              )
]

setup(
    name="FIRD",
    ext_modules=cythonize(ext_modules),
    packages=find_packages()
)
