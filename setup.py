from setuptools import setup, find_packages
from codecs import open
import os

# Get the directory where this current file is saved
here = os.path.abspath(os.path.dirname(__file__))

# Read the README.md file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read the version from version.txt
version_file = os.path.join(here, 'ads_base', 'version.txt')
with open(version_file, 'r') as f:
    __version__ = f.read().strip()

# Read requirements from requirements.txt
requirements = []
requirements_file = os.path.join(here, 'requirements.txt')
if os.path.exists(requirements_file):
    with open(requirements_file) as f:
        requirements = f.read().splitlines()

setup(
    name='ads_base',
    python_requires='>=3.11',
    version=__version__,
    description='Python tool for automatic axon and myelin segmentation',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Important for PyPI
    url='https://github.com/neuropoly/axondeepseg',
    author='NeuroPoly Lab, Polytechnique Montreal',
    author_email='neuropoly@googlegroups.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='axon myelin segmentation microscopy',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    package_dir={'ads_base': 'ads_base'},
    package_data={
        "ads_base": [
            'models/default_SEM_model/*',
            'models/default_TEM_model/*',
            'models/default_BF_model/*',
            'data_test/*',
            'version.txt',  # Include version.txt in the package
        ],
    },
    include_package_data=True,  # Ensure package_data files are included
    install_requires=requirements,  # Use the requirements from requirements.txt
    entry_points={
        'console_scripts': [
           'download_model = ads_base.download_model:main',
           'download_tests = ads_base.download_tests:main',
           'axondeepseg = ads_base.segment:main',
           'axondeepseg_test = ads_base.integrity_test:main', 
           'axondeepseg_morphometrics = ads_base.morphometrics.launch_morphometrics_computation:main'
        ],
    },
)