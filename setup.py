from setuptools import setup, find_packages


with open("README.md", "r") as f:
    description = f.read()

setup(
    name="scDiffusion",
    version="0.8",
    description='scDiffusion(Single-Cell graph neural Diffusion) is a physics-informed graph generative model to do scRNA-seq analysis. scDiffusion investigates cellular dynamics utilizing an attention-based neural network.',
    author="Yuchen Liu",
    author_email="ycliu137@bu.edu",
    url='https://github.com/CZCBLab/scDiffusion/tree/main',
    packages=find_packages(),
    install_requires=['leidenalg>=0.8.10', 'matplotlib>=3.9.2', 'networkx>=3.4.2', 'numpy==1.26.4', 'pandas>=2.2.3', 'python_igraph>=0.9.9', 'python_louvain>=0.16',
                      'scanpy', 'scikit_learn>=1.0.2', 'scipy>=1.14.1', 'umap_learn>=0.5.2', 'torch==1.13.1', 'torchvision==0.14.1', 
                      'torchaudio==0.13.1'],
    python_requires=">=3.9, <3.11",
    keywords=['single-cell', 'diffusion'],
    license='MIT',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    long_description=description,
    long_description_content_type="text/markdown",
)
