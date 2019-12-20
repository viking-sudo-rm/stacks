#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/stacks/configs/agree-dmg.jsonnet

encoder=dmg  # Should probably only use this script with DMG(-like) stuff.
lstm=true  # Use an LSTM cell in the stack controller.

if [ $lstm == "true" ]
then
    dir=/tmp/willm/agree-${encoder}-lstm
else
    dir=/tmp/willm/agree-$encoder
fi

ENCODER=$encoder LSTM=$lstm allennlp train $conf -s $dir --include-package=src
