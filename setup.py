from setuptools import setup, find_packages
from setuptools.command.develop import develop
from codecs import open
from os import path,system

from AxonDeepSeg.ads_utils import download_data
import AxonDeepSeg


# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

req_path = path.join(here, 'requirements.txt')
with open(req_path, "r") as f:
    install_reqs = f.read().strip()
    install_reqs = install_reqs.split("\n")


class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):

        develop.run(self)

        # Download Models from OSF storage
        url_TEM_model = "https://osf.io/2hcfv/?action=download"  # URL of TEM model hosted on OSF storage
        url_SEM_model = "https://osf.io/rdqgb/?action=download"  # URL of SEM model hosted on OSF storage
        if (not download_data(url_TEM_model) and not download_data(url_SEM_model)) == 1:
            print('Data downloaded and unzipped succesfully.')
        else:
            print('ERROR: Data was not succesfully downloaded and unzipped- please check your link and filename and try again.')
        system("mv default* AxonDeepSeg/models/") # Migrate models from current directory to models directory



setup(
    name='AxonDeepSeg',
    python_requires='>=3.6, <3.7',
    version=AxonDeepSeg.__version__,
    description='Python tool for automatic axon and myelin segmentation',
    long_description=long_description,
    url='https://github.com/neuropoly/axondeepseg',
    author='NeuroPoly Lab, Polytechnique Montreal',
    author_email='neuropoly@googlegroups.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],

    keywords='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=install_reqs,
    package_dir={'AxonDeepSeg': 'AxonDeepSeg'},
    package_data={
        "AxonDeepSeg": ['models/default_SEM_model_v1/*',
                        'models/default_TEM_model_v1/*',
                        'data_test/*'],
    },
    extras_require={
        'docs': ['sphinx>=1.6',
                 'sphinx_rtd_theme>=0.2.4'],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'axondeepseg = AxonDeepSeg.segment:main','axondeepseg_test = AxonDeepSeg.integrity_test:integrity_test'
        ],
    },
    cmdclass={
        'develop': PostDevelopCommand,
    },

)
