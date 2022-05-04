#!/bin/bash
export task=qnli
# sbatch finetune_glue_seq-len.slurm
sbatch finetune_glue_mp0_15.slurm
# sbatch finetune_glue_mp0_4.slurm