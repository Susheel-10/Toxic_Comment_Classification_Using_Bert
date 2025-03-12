from setuptools import setup, find_packages

setup(
    name="toxic_comment_detector",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'nltk',
        'fastapi',
        'uvicorn',
        'scikit-learn',
        'numpy',
        'pandas',
        'tqdm',
        'pydantic'
    ]
) 