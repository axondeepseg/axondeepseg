FROM continuumio/anaconda3

USER root

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
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
        curl\
        wget \
        gfortran \
        cmake \
        bsdtar \
        rsync \
        imagemagick \
        \
        gnuplot-x11 \
        libopenblas-base \
        \
        ttf-dejavu && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN cd $HOME/work;\
    pip install --upgrade pip;\
    pip install sos sos-notebook scipy plotly dash dash_core_components dash_html_components dash_dangerously_set_inner_html dash-renderer flask==0.12.2;\
    python -m sos_notebook.install;\
    git clone --single-branch -b dev_mikula_docker https://github.com/neuropoly/axondeepseg; \
    cd axondeepseg;\
    sudo ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh;\
    conda create -n ads_venv python=3.6;\
    conda activate ads_venv;\
    pip install -e .;

WORKDIR $HOME/work/t1_notebooks

USER $NB_UID
