#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="doe",
    version="0.1",
    description="Decoder Only Model Experiment",
    author="Mizaimao",
    packages=find_packages(include=["doe", "doe.*"]),
)
