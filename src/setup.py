from setuptools import setup, find_namespace_packages

setup(
    name="opentelemetry-instrumentation-langchain-v2",
    version="0.1",

    # This is critical for namespace packages
    packages=find_namespace_packages(where="src", include=["opentelemetry.*"]),
    package_dir={"": "src"},

    install_requires=[
        "opentelemetry-api>=1.0.0",
        "opentelemetry-sdk>=1.0.0",
        "langchain>=0.0.1",
    ],

    # Include non-Python files
    include_package_data=True,
)
