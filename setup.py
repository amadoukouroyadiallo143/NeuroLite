import os
from setuptools import setup, find_packages

setup(
    name='neurolite',
    version='0.1.0',
    packages=find_packages(include=['neurolite', 'neurolite.*']),
    author='Amadou Kouro Diallo', # J'ai mis votre nom d'utilisateur, vous pouvez changer
    author_email='your.email@example.com', # À remplacer si vous le souhaitez
    description='Une bibliothèque IA légère et modulaire pour des modèles polyvalents.',
    long_description=open('README.md', encoding='utf-8').read() if os.path.exists('README.md') else '',
    long_description_content_type="text/markdown",
    # url='https://github.com/your_username/NeuroLite', # À remplacer si vous avez un repo
    install_requires=[
        'numpy>=1.19.0',
        'torch>=1.12.0',
        'einops>=0.3.0',
        'mmh3>=3.0.0',
        'bitarray>=2.3.0',
    ],
    python_requires='>=3.8',
)
