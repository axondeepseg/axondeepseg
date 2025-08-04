#!/bin/bash
#SBATCH --account=def-jcohen
#SBATCH --mem=60000
#SBATCH -c 4
#SBATCH --gpus=1
#SBATCH --time=0-03:00           # time (DD-HH:MM)
#SBATCH --mail-user=murielle.mardenli@gmail.com
#SBATCH --mail-type=ALL

#SBATCH --output=output.log
#SBATCH --error=error.log

module load StdEnv/2020

module load python/3.11

# Create virtual environment if non existent
VENV_DIR=$HOME/envs/myenv
if [ ! -d "$VENV_DIR" ]; then
    python -m venv $VENV_DIR
fi

source $VENV_DIR/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

pip install axondeepseg --no-build-isolation --no-cache-dir

# Run segmentation
axondeepseg -i data_axondeepseg_sem/sub-rat1/micr/sub-rat1_sample-data1_SEM.png
