#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/stacks/configs/code-gen.jsonnet

encoder=kpop
dataset=tensorflow
names=trues

if [ "$names" = "true" ]; then
    dir=/tmp/willm/stacks/lm/${dataset}/${encoder}-names
else
    dir=/tmp/willm/stacks/lm/${dataset}/${encoder}
fi

DATASET=$dataset ENCODER=$encoder NAMES=$names allennlp train $conf -s $dir --include-package=src
