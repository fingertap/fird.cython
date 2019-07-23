from setuptools import setup, Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("fird",
              sources=["fird/fird.pyx"],
              libraries=["m"]  # Unix-like specific
              )
]

setup(
    name="FIRD",
    ext_modules=cythonize(ext_modules)
)
