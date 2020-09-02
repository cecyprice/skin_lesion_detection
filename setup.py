from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'gcsfs==0.6.0',
    'google-cloud-storage==1.26.0',
    'pip>=9',
    'setuptools>=26',
    'wheel>=0.29',
    'numpy==1.18.4',
    'pandas==0.24.2',
    'pytest',
    'coverage',
    'flake8',
    'black',
    'yapf',
    'python-gitlab',
    'twine',
    'Keras>=2.3.0',
    'tensorflow>=2.3.0',
    'imblearn',
    'keras-tuner',
    'colored',
    'scikit-learn>=0.23.0',
    'matplotlib==3.2.1',
    'imageio',
    'Pillow',
    'glob3',
    'joblib',
    'google.cloud',
    'datetime',
    'streamlit==0.65.2',
    'plotly==4.6.0',
    'scipy==1.2.2',
    'requests==2.23.0',
    'bs4==0.0.1',
    'termcolor',
    'h5py']

#with open('requirements.txt') as f:
#    content = f.readlines()
#requirements = [x.strip() for x in content if 'git+' not in x]


setup(name='skin_lesion_detection',
      version="1.0",
      install_requires=REQUIRED_PACKAGES,
      description="Project Description",
      packages=find_packages(),
      test_suite = 'tests',
      include_package_data=True,
      scripts=['scripts/skin_lesion_detection-run'],
      zip_safe=False)
