#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/stacks/configs/eval.jsonnet

encoder=kpop-lstm
prior=0.5
dir=/tmp/willm/stacks/eval/${encoder}-${prior}
ENCODER=$encoder PRIOR=$prior allennlp train $conf -s $dir --include-package=src
