from setuptools import setup, find_packages

setup(
    name="ingestion-agent",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "aioboto3>=12.3.0",
        "boto3>=1.34.34",
        "aiofiles>=23.2.1",
        "Pillow>=10.2.0",
        "numpy>=1.26.4",
        "dvc[s3]>=3.48.4",
        "opencv-python>=4.9.0.80",
        "albumentations>=1.4.0",
        "aiohttp>=3.9.3",
        "python-json-logger>=2.0.7",
        "prometheus-client>=0.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.2",
            "pytest-asyncio>=0.23.5",
            "pytest-cov>=4.1.0",
            "moto>=5.0.2",
            "localstack-client>=2.5",
            "black>=24.2.0",
            "flake8>=7.0.0",
            "mypy>=1.8.0",
            "isort>=5.13.2",
        ]
    },
)
