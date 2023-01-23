# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 12:51:32 2022

@author: 318596
"""

import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hydropop",
    version="1.0",
    author="Jon Schwenk",
    author_email="jschwenk@lanl.gov",
    description="Package to create hydropop scaling units",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lanl/hydropop",
    project_urls={
        "Bug Tracker": "https://github.com/lanl/hydropop/issues",
    },
    # scripts=["rabpro/cli/rabpro"],
    python_requires='>=3',
    install_requires=[
        "gdal",
        "numpy",
        "geopandas>=0.7.0",
        "scikit-image",
        "pyproj",
        "shapely",
        "requests",
        "appdirs",
        "earthengine-api",
        "gdown",
        "tqdm",
        "beautifulsoup4",
    ],
    # extras_require={
    #     "test": ["pytest>=3"],
    # },
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
)