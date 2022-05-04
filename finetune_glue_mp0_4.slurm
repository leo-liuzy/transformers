#!/bin/bash
#
#SBATCH --job-name=finetune.roberta.
#SBATCH --output=/private/home/zeyuliu/masking_strategy/transformers/slurm_output/finetune.stdout.%j
#SBATCH --error=/private/home/zeyuliu/masking_strategy/transformers/slurm_output/finetune.stderr.%j
#SBATCH --ntasks=1
#SBATCH --ntasks=1
##SBATCH --unbuffered=true
##SBATCH --cpu-bind=map_ldom:0,0,0,0,1,1,1,1
#SBATCH --open-mode=append
#SBATCH --mem=60G 
#SBATCH --time=4320
#SBATCH --signal=B:USR1@180

#SBATCH --constraint=volta32gb
#SBATCH --partition learnaccel
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
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

task=$task
config_dir=/private/home/zeyuliu/masking_strategy/transformers/finetuning_hf

masking=mp0.4
model_path=/checkpoint/zeyuliu/en_dense_lm/masking_strategies/roberta.base.faststatsync.me_fp16.cmpltsents.mp0.4.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf4.mu500000.s1.ngpu64/checkpoint_last.pt

date 
echo $masking
echo $task

python roberta_run_glue.py \
  -task $task \
  -masking $masking \
  -finetune_config_dir $config_dir \
  -finetune_output_dir results/roberta-base \
  -path_to_pretrained_checkpoint $model_path