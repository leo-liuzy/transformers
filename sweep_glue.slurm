#!/bin/bash
#
#!/bin/bash

#SBATCH --constraint=volta32gb
#SBATCH --partition learnaccel
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10

#SBATCH --array=1-20
#SBATCH --export=ALL
#SBATCH --output slurm/sweep_%j_%a.log
#SBATCH --output slurm/sweep_%j_%a.stderr

#SBATCH --job-name=training
#SBATCH --nodes=1
##SBATCH --unbuffered=true
##SBATCH --cpu-bind=map_ldom:0,0,0,0,1,1,1,1
#SBATCH --open-mode=append
#SBATCH --mem=480G 
#SBATCH --time=1440
#SBATCH --signal=B:USR1@180

wandb agent --count 3 $@
