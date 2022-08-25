task=rte
config_dir=/private/home/zeyuliu/masking_strategy/transformers/finetuning_hf_8gpu_eval_per_epoch

masking=mp0.5
model_path=/checkpoint/zeyuliu/en_dense_lm/masking_strategies/roberta.base.faststatsync.me_fp16.cmpltsents.mp0.5.roberta_base.tps512.adam.fp16adam.b2_0.98.eps1e-06.cl0.0.lr0.0006.wu24000.dr0.1.atdr0.1.wd0.01.ms32.uf4.mu500000.s1.ngpu64/checkpoint_31_300000.pt

date 
echo $masking
echo $task

python roberta_run_glue.py \
  -task $task \
  -masking $masking \
  -finetune_config_dir $config_dir \
  -finetune_output_dir results/roberta-base-8gpu \
  -path_to_pretrained_checkpoint $model_path