FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y build-essential \
    autoconf \
    libtool \
    pkg-config \
    doxygen \
    wget \
    git \
    python3 \
    python3-dev \
    python3-pip

# Yes we are cheating with xmltodict :)
RUN pip3 install ipython xmltodict && \
    git clone https://github.com/llnl/shroud && \
    cd shroud && \
    pip3 install .
WORKDIR /code
ADD . /code
