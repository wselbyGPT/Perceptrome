from setuptools import find_packages, setup


setup(
    name="perceptrome",
    version="0.0.0",
    description="Perceptrome CLI",
    packages=find_packages(exclude=("tests", "tests.*")),
    install_requires=["numpy", "requests", "torch", "tensorboard", "PyYAML", "PySide6>=6.6"],
    entry_points={
        "console_scripts": [
            "perceptrome=perceptrome.cli_main:main",
        ]
    },
)
