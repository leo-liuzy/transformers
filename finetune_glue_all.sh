#!/bin/bash
# sbatch finetune_glue_seq-len.slurm
for task in mnli qnli sst2 qqp
do 
    export task=$task
    sbatch finetune_glue_mp0_15.slurm
done
# sbatch finetune_glue_mp0_4.slurm