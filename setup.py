from setuptools import setup, find_packages

setup(
    name="aillmclean",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "requests>=2.28.0",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI-powered Python library for context-aware data cleaning using local LLMs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llmclean",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
)
