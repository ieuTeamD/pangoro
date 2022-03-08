
from setuptools import setup, find_packages

setup(
    name='pangoro',
    version='0.29',
    packages=find_packages(exclude=['tests*']),
    license='MIT',
    description='pangoro python package',
    long_description=open('README.txt').read(),
    install_requires=['numpy','pandas','sklearn','seaborn','matplotlib'],
    url='https://github.com/ieuTeamD',
    author='Team D',
    author_email='ieuTeamD@gmail.com'
)
