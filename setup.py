from setuptools import setup

setup(
    name="predictor",
    version="1.0.0",
    description="...",
    author="Andreas Burger",
    author_email="me@andreas-burger.com",
    url = "https://github.com/BurgerAndreas",
    # point to folders that you want to be able to import
    packages=["predictor"],
    install_requires=[
        'ase>=3.22.1',
        'numpy<=1.25.0',
        'asap3>=3.12.8',
        'torch>=1.10',
        'scipy',
    ]
)
