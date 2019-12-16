#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate bert-parsing
conf=/home/willm/code/stacks/configs/dyck-lm.jsonnet

encoder=stack
dir=/tmp/willm/dyck-$encoder
ENCODER=$encoder allennlp train $conf -s $dir --include-package=src
