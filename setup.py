"""
setup.py

Installation script for Parkinson's Insight Engine (PIE).
"""

import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="parkinsons-insight-engine",
    version="0.0.1",
    author="Cameron Hamilton and the Data Modality and Methodology Task Force",
    author_email="cameron@allianceai.co",
    description="A data preprocessing and analysis pipeline for Parkinson's research (PIE)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MJFF-ResearchCommunity/PIE",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
