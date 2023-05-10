FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get upgrade -y && apt-get install -y --no-install-recommends\
    nano \
    curl \
    apt-utils \
    ssh \
    tree \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    llvm \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libbz2-dev \
    libgdbm-dev \
    libnss3-dev \
    libreadline-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.8.5/Python-3.8.5.tgz
RUN tar -xvzf Python-3.8.5.tgz

WORKDIR Python-3.8.5
RUN ./configure
RUN make install

WORKDIR /
RUN rm -rf Python-3.8.5 \
    rm Python-3.8.5.tgz

RUN echo "alias python=python3.8" >> ~/.bashrc

RUN python3 -m pip install --upgrade pip
RUN wget "https://raw.githubusercontent.com/krchanyang/PolyMed/main/requirements.txt"
RUN pip install -r requirements.txt
