import os
from setuptools import setup, find_packages

setup(
    name='team85-voice-recognition',
    description='Voice Recognition',
    packages=find_packages(),
    version=os.environ.get('BUILDNUMBER')
)