#!/bin/bash
#PBS -N Network_ROB
#PBS -A rrp-355-aa
#PBS -l walltime=43200
#PBS -l nodes=1:gpus=1

cd ${PBS_O_WORKDIR}

source /home/pilou/buildenv/bin/activate

module load compilers/gcc/4.8.5 compilers/java/1.8 apps/buildtools \
                            cuda/7.5 libs/cuDNN/5 compilers/swig apps/git apps/bazel/0.4.3

python /scratch/rrp-355-aa/pilou/working_rep/axondeepseg/AxonDeepSeg/guide_robert.py
