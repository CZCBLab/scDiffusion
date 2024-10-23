FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Set timezone
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN apt-get update && apt-get install -yq software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update && apt-get install -yq --no-install-recommends \
    # Python packages
    python3.9-dev python3-pip python3.9-tk python3.9-distutils && \	
    # clean up
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade --no-cache \
    leidenalg==0.8.10 matplotlib==3.5.1 networkx==2.6.3 pandas==1.3.5 python_igraph==0.9.9 python_louvain==0.16 scikit_learn==1.0.2 scipy==1.7.3 umap_learn==0.5.2 ipywidgets

# Install PyTorch CUDA
RUN pip install --upgrade --no-cache \
    torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

# Install scanpy
RUN pip install --upgrade --no-cache scanpy numpy==1.26.4

# Install Jupyter
RUN pip install --upgrade --no-cache jupyterlab notebook

RUN mkdir /scDiffusion
WORKDIR /scDiffusion
COPY . .

CMD ["jupyter notebook --ip='0.0.0.0' --allow-root"]

EXPOSE 8888