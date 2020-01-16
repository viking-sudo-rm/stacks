#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/stacks/configs/num-pops-lm.jsonnet

dataset=penn
encoder=kpop-lstm
pops_weight=0.0
prior_weight=1.0

dir=/tmp/willm/stacks/num-pops-lm/${dataset}/${encoder}-0,1-rev

POPS=$pops_weight PRIOR=$prior_weight DATASET=$dataset ENCODER=$encoder allennlp train \
    $conf -s $dir --include-package=src
