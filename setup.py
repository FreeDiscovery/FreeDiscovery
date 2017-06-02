#!/usr/bin/python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import re


# a define the version sting inside the package
# see https://stackoverflow.com/questions/458550/standard-way-to-embed-version-into-python-package
VERSIONFILE = "freediscovery/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    version = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

with open('README.rst', 'rt') as fh:
    LONG_DESCRIPTION = fh.read()

with open('requirements.txt', 'rt') as fh:
    REQUIREMENTS = fh.read().splitlines()

REQUIREMENTS_CORE = REQUIREMENTS[:5]

setup(name='freediscovery',
      version=version,
      description='Open source software for E-Discovery '
                  'and Information Retrieval',
      author='Grossman Labs',
      url="https://github.com/FreeDiscovery/FreeDiscovery",
      license='BSD',
      packages=find_packages(),
      include_package_data=True,
      long_description=LONG_DESCRIPTION,
      python_requires='>=3.5',
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

          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',

          'Operating System :: OS Independent',

          'Topic :: Scientific/Engineering :: Artificial Intelligence',
          'Topic :: Text Processing :: General'],
      extras_require={'all': REQUIREMENTS,
                      'core': REQUIREMENTS_CORE},
      keywords='information-retrieval machine-learning text-classification')

