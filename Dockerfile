FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

MAINTAINER enoken enoken@ucl.nuee.nagoya-u.ac.jp

RUN apt-get update && apt-get install -y --no-install-recommends \
		build-essential \
		git \
		curl \
		ca-certificates \
		libjpeg-dev \
		libpng-dev \
		python-pip && \
	rm -rf /var/lib/apt/lists/*

ENV PYTHON_VERSION=3.6
RUN curl -L -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
     chmod +x ~/miniconda.sh && \
     ~/miniconda.sh -b -p /opt/conda && \     
     rm ~/miniconda.sh && \
     /opt/conda/bin/conda install conda-build && \
     /opt/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl && \
     /opt/conda/bin/conda clean -ya 
ENV PATH opt/conda/bin:/opt/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH
RUN conda install --name pytorch-py$PYTHON_VERSION -c pytorch magma-cuda80 && /opt/conda/bin/conda clean -ya
RUN conda install --name pytorch-py$PYTHON_VERSION pytorch torchvision cuda80 -c pytorch && /opt/conda/bin/conda clean -ya

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

WORKDIR /workspace

RUN git clone https://github.com/enomotokenji/pytorch-Neural-Style-Transfer.git

RUN chmod -R a+w /workspace
