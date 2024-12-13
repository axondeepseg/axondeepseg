from setuptools import setup, find_packages
from setuptools.command.develop import develop
from subprocess import check_call

from codecs import open
import os


# Get the directory where this current file is saved
here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

version_file = os.path.abspath(os.path.join(__file__, os.pardir, "AxonDeepSeg", "version.txt"))
with open(version_file, 'r') as f:
    __version__ = f.read().rstrip()

setup(
    name='AxonDeepSeg',
    python_requires='>=3.11',
    version=__version__,
    description='Python tool for automatic axon and myelin segmentation',
    long_description=long_description,
    url='https://github.com/neuropoly/axondeepseg',
    author='NeuroPoly Lab, Polytechnique Montreal',
    author_email='neuropoly@googlegroups.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],

    keywords='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    package_dir={'AxonDeepSeg': 'AxonDeepSeg'},
    package_data={
        "AxonDeepSeg": ['models/default_SEM_model/*',
                        'models/default_TEM_model/*',
                        'models/default_BF_model/*'
                        'data_test/*'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
           'download_model = AxonDeepSeg.download_model:main',
           'download_tests = AxonDeepSeg.download_tests:main',
           'axondeepseg = AxonDeepSeg.segment:main',
           'axondeepseg_test = AxonDeepSeg.integrity_test:main', 
           'axondeepseg_morphometrics = AxonDeepSeg.morphometrics.launch_morphometrics_computation:main'
        ],
    },

)
