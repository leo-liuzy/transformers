#!/bin/bash
# sbatch finetune_glue_seq-len.slurm
for task in qnli sst2 # mnli qnli sst2 qqp
do 
    export task=$task
    # sbatch finetune_glue_mp0_15.slurm
    # sbatch finetune_glue_mp0_4.slurm
    # sbatch finetune_glue_mp0_5.slurm
    # sbatch finetune_glue_seq-len_0_1_0_9.slurm
    sbatch finetune_glue_seq-len.slurm
done