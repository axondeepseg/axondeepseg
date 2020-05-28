FROM jupyter/datascience-notebook

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        emacs \
        git \
        inkscape \
        jed \
        libsm6 \
        libxext-dev \
        libxrender1 \
        lmodern \
        netcat \
        unzip \
        nano \
        curl \
        wget \
        cmake \
        bsdtar  \
        rsync \
        gnuplot-x11 \
        libopenblas-base \
        python3-dev \
        ttf-dejavu && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN git clone --single-branch -b docker https://github.com/neuropoly/axondeepseg; \
    cd axondeepseg;\
    pip install -e .;\
    chmod -R 777 $HOME/work/axondeepseg

WORKDIR $HOME/work/axondeepseg

USER $NB_UID
