#!/bin/bash
#export HOME=/home/epic
#cd /home/batman/epic/LWA_EPIC/build
eval "$(/home/epic/anaconda/condabin/conda shell.bash hook)"
conda activate epic310

#cd src
addr="$1"
python ./src/run.py start -a "${addr}"