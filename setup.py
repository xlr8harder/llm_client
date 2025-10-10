from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Single-source the version from the package
try:
    from llm_client import __version__ as package_version
except Exception:
    package_version = "0.0.0"

setup(
    name="llm_client",
    version=package_version,
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
        "urllib3>=1.26.0",
    ],
)
