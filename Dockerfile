FROM jupyter/datascience-notebook:python-3.7.6

USER root

RUN cd $HOME/work;\
    git clone https://github.com/neuropoly/axondeepseg.git; \
    cd axondeepseg; \
    pip install -e . ; \
    chmod -R 777 $HOME/work/axondeepseg;

WORKDIR $HOME/work/axondeepseg

USER $NB_UID
