from __future__ import absolute_import
import os
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import setup, find_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sklearn-container',
    version='1.0',
    description='Example container with scikit-learn.',
    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    long_description=read('README.rst'),
    author='Horacio Ferro',
    license='Apache License 2.0',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11'
    ],
    install_requires=read('requirements.txt'),
    entry_points={
        'console_scripts': 'serve=sklearn_container.serving:serving_entrypoint'
    },
    python_requires='>=3.8'
)