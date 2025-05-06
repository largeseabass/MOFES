from setuptools import setup, find_packages

setup(
    name='mofes',
    version='0.1.0',
    description='MOFES: Mosquito Forecasting and Estimation Simulator using FEniCSx',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    author='Liting Huang',
    author_email='litinghuang42@gmail.com',
    url='https://github.com/largeseabass/MOFES',
    license='MIT',
    project_urls={
        'Source': 'https://github.com/largeseabass/MOFES',
    },
    packages=find_packages(exclude=["tests*", "docs*", "notebooks*"]),
    python_requires='>=3.8',
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
        "xarray",
        "netCDF4",
        "scikit-learn",
        "pyvista",
        "opencv-python",
        "ipyparallel",
        "tqdm",
        "loguru",
    ],
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            'mofes-run = mofes.cli:main',
        ]
    },
)
