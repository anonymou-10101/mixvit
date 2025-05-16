FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu18.04
WORKDIR /app/

# Install Requirements
RUN apt-get update && apt-get install -y ffmpeg libturbojpeg wget git screen\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Conda
RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN chmod +x Miniconda3-latest-Linux-x86_64.sh
RUN ./Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda
    
ENV PATH=/miniconda/bin:$PATH
    
RUN /miniconda/bin/conda install -y conda-build \
    && /miniconda/bin/conda create -y --name mixvit python=3.9 \
    && /miniconda/bin/conda clean -ya
    
ENV CONDA_DEFAULT_ENV=mixvit
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false
    
RUN conda install -y ipython

RUN conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
RUN conda clean -ya

ENV PYTHONPATH=/miniconda/envs/mixvit/lib/python3.9/site-packages:$PYTHONPATH

# Install ffmpeg with conda directly and the libx264 will be built automatically.
RUN conda install ffmpeg

RUN pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html

# Install cython ....
RUN apt-get update && apt-get install -y gcc build-essential
RUN pip install --upgrade pip setuptools wheel cython==0.29.33 numpy==1.24.4 yapf==0.31.0
RUN pip install mmpycocotools
RUN pip install pywavelets

# Install Apex
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=$CUDA_HOME/bin:$PATH
# ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

RUN git clone https://github.com/NVIDIA/apex.git
RUN cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# This is for MMDetection
RUN git clone https://github.com/SwinTransformer/Swin-Transformer-Object-Detection.git 
RUN cd Swin-Transformer-Object-Detection && pip install -v -e .
RUN pip install mmdet

ENV PYTORCH_CUDA_ALLOC_CONF=""