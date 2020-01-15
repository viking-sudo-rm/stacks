#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/stacks/configs/yogatama-lm.jsonnet

dataset=penn
encoder=kpop-lstm
dir=/tmp/willm/stacks/yoga2-lm/${dataset}/$encoder

DATASET=$dataset ENCODER=$encoder allennlp train $conf -s $dir --include-package=src
