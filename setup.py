from setuptools import setup, find_packages

setup(
    name="bdpa-clickflow",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn"
    ],
)
