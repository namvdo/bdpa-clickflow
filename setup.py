from setuptools import setup

setup(
    name="bdpa-clickflow",
    version="0.2.0",
    packages=["utils", "pipeline"],
    install_requires=[
        "streamlit",
        "pandas",
        "numpy",
        "matplotlib",
        "seaborn",
        "scikit-learn"
    ],
)
