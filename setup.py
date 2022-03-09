
from setuptools import setup, find_packages

setup(
    name='pangoro',
    version='0.32',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='pangoro python package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',

    install_requires=['numpy','pandas','sklearn','seaborn','matplotlib'],
    url='https://github.com/ieuTeamD',
    author='Team D',
    author_email='ieuTeamD@gmail.com'
)
