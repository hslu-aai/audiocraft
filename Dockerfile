FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get install -y libtcmalloc-minimal4
RUN apt-get install -y git
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime \
    && echo "Etc/UTC" > /etc/timezone \
    && apt-get update \
    && apt-get upgrade -y \
    && apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended xzdec dvipng cm-super -y \
    && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get install -y \
        apt-transport-https \
        libtcmalloc-minimal4 \
        libomp-dev \
        sox \
        git \
        gcc \
        g++ \
        python3-dev \
        ffmpeg \
        libsm6 \
        libxext6 \
    && apt-get clean

RUN pip install --upgrade pip

WORKDIR /workspace

RUN mkdir /assets

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt --upgrade --no-cache-dir
RUN pip install -r 

COPY . /workspace/
RUN git config --global --add safe.directory '*'

ENV PYTHONPATH "$PYTHONPATH:./"
RUN export USER="$(whoami)"
