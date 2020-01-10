#!/bin/bash
#SBATCH --mail-user=willm@allenai.org
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output=/net/nfs.corp/allennlp/slurm/willm/%J.out
#SBATCH --gpus=1

source /home/willm/anaconda3/bin/activate allennlp
conf=/home/willm/code/stacks/configs/ud-parsing.jsonnet

controller=dmg-lstm
hard=true

if [ "$hard" = "true" ]; then
    dir=/tmp/willm/stacks/ud-${controller}-hard
else
    dir=/tmp/willm/stacks/ud-$controller
fi

CTYPE=$controller HARD=$hard allennlp train $conf -s $dir --include-package=src
