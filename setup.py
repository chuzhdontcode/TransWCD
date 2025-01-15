#!/usr/bin/env python

from setuptools import setup, find_packages

# Read the requirements from requirements.txt
def parse_requirements(filename):
    with open(filename, "r") as file:
        return [line.strip() for line in file if line.strip() and not line.startswith("#")]

setup(
    name="transwcd",
    version="0.0.0",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url="",
    install_requires=parse_requirements("transwcd/requirements/base.txt"),
    packages=find_packages(),
)
