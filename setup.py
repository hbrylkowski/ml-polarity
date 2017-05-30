from setuptools import setup, find_packages

setup(name='polarity',
  version='0.1',
  packages=find_packages(),
  description='example of polarity score nn with keras runnable on gcloud ml-engine',
  author='Hubert Brylkowski',
  author_email='hubert@brylkowski.com',
  install_requires=[
      'keras',
      'h5py'
  ],
  zip_safe=False)
