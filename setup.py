# -*- coding: utf-8 -*-
import sys

from freediscovery._version import __version__
from setuptools import setup, find_packages


with open('README.rst', 'rb') as fh:
    LONG_DESCRIPTION = fh.read().decode('utf-8')

with open('requirements.txt', 'rt') as fh:
    REQUIREMENTS = fh.read().splitlines()

with open('requirements_engine.txt', 'rt') as fh:
    REQUIREMENTS_ENGINE = fh.read().splitlines()


setup(name='freediscovery',
      version=__version__,
      description='Open source software for E-Discovery '
                  'and Information Retrieval',
      author='Grossman Labs',
      url="https://github.com/FreeDiscovery/FreeDiscovery",
      license='BSD',
      packages=find_packages(),
      include_package_data=True,
      long_description=LONG_DESCRIPTION,
      entry_points={
          'console_scripts': [
              'freediscovery = freediscovery.__main__:main'
          ]
      },
      classifiers=[
          'Development Status :: 4 - Beta',

          'Intended Audience :: Information Technology',
          'Intended Audience :: Legal Industry',

          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',

          'Operating System :: OS Independent',

          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Text Processing :: General'],
      extras_require={'engine': REQUIREMENTS_ENGINE,
                      'core': REQUIREMENTS},
      keywords='information-retrieval machine-learning text-classification')
