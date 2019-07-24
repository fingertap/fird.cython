import os
import numpy as np
from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize

NAME = 'fird'

AUTHOR = 'Han Zhang'

AUTHOR_EMAIL = 'zh950713@gmail.com'

GITHUB = 'https://www.github.com/fingertap/fird.cython'

VERSION = '0.0.1'

DESCRIPTION = ('FIRD: Fraud Detection by Simultaneous '
               'Feature Selection and Clustering')


def long_desc():
    return open(os.path.join(os.path.dirname(__file__), 'README.md')).read()


ext_modules = [
    Extension("fird.fird",
              sources=["fird/fird.pyx"],
              libraries=["m"]  # Unix-like specific
              )
]

setup(
    name=NAME,
    version=VERSION,
    ext_modules=cythonize(ext_modules),
    packages=find_packages(),
    description=DESCRIPTION,
    long_description=long_desc(),
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=GITHUB,
    license='MIT',
    include_dirs=[np.get_include()],
    install_requires=['Cython']
)
