import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ecopop",
    version="1.0",
    author="Jon Schwenk",
    author_email="jschwenk@lanl.gov",
    description="Package to create ecopop scaling units",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lanl/ecopop",
    project_urls={
        "Bug Tracker": "https://github.com/lanl/ecopop/issues",
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
    extras_require={
         "test": ["pytest>=3"],
         "docs": ["sphinx", "sphinx_rtd_theme"]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        # "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
)
