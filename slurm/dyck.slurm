#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/stacks/configs/dyck-lm.jsonnet

encoder=stack
num_ops=5
dir=/tmp/willm/dyck${num_ops}-$encoder
NUM_OPS=$num_ops ENCODER=$encoder allennlp train $conf -s $dir --include-package=src
