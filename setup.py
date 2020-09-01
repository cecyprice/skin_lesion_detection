from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'streamlit==0.65.2',
    'numpy==1.18.4',
    'pandas==0.24.2',
    'tensorflow>=2.3.0',
    'matplotlib==3.2.1',
    'plotly==4.6.0',
    'scipy==1.2.2',
    'requests==2.23.0',
    'bs4==0.0.1',
    'termcolor',
    'keras-tuner',
    'glob',
    'sklearn',
    'imageio',
    'PIL',
    'imblearn',
    'warnings',
    'json',
    'joblib',
    'google.cloud',
    'google.oauth2'
    'datetime'
    ]


setup(name='skin_lesion_detection',
      version="1.0",
      install_requires=REQUIRED_PACKAGES,
      description="Project Description",
      packages=find_packages(),
      test_suite = 'tests',
      include_package_data=True,
      scripts=['scripts/skin_lesion_detection-run'],
      zip_safe=False)
