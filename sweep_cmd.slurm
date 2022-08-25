#!/bin/bash
#
#SBATCH --job-name=finetune.roberta.
#SBATCH --output=/private/home/zeyuliu/masking_strategy/transformers/slurm_output/finetune.stdout.%j
#SBATCH --error=/private/home/zeyuliu/masking_strategy/transformers/slurm_output/finetune.stderr.%j
#SBATCH --nodes=1
##SBATCH --unbuffered=true
##SBATCH --cpu-bind=map_ldom:0,0,0,0,1,1,1,1
#SBATCH --open-mode=append
#SBATCH --mem=480G 
#SBATCH --time=4320
#SBATCH --signal=B:USR1@180

#SBATCH --constraint=volta32gb
#SBATCH --partition learnaccel
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=10

trap_handler () {
   echo "Caught signal: " $1
   # SIGTERM must be bypassed
   if [ "$1" = "TERM" ]; then
       echo "bypass sigterm"
   else
     # Submit a new job to the queue
     echo "Requeuing " $SLURM_JOB_ID
     scontrol requeue $SLURM_JOB_ID
   fi
}


# Install signal handler
trap 'trap_handler USR1' USR1
trap 'trap_handler TERM' TERM

wandb agent leo-liuzy/roberta_seq-len_0_2_0_8_squad/tn15od4k
