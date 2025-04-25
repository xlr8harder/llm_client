from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm_client",
    version="0.1.0",
    author="",
    author_email="",
    description="A clean, modular client library for interacting with various LLM providers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xlr8harder/llm_client",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "requests>=2.25.0",
    ],
)
