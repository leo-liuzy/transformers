# Created by xunannancy at 2022/4/30
import json
import logging
import os
import random
import shutil
from dataclasses import dataclass, field
from typing import Optional
import uuid

import numpy as np
import transformers
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    RobertaTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaForSequenceClassification,
    EvalPrediction
)
from transformers.models.bert.modeling_bert import (
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertSelfAttention,
    BertSelfOutput,
)
from transformers.trainer_utils import is_main_process
logger = logging.getLogger(__name__)
import argparse
import torch
import pathlib
from pdb import set_trace as bp

try:
    from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
    from fairseq.modules import TransformerSentenceEncoderLayer
except:
    print('current env does not have package fairseq...')

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in task_to_keys.keys():
                raise ValueError(
                    "Unknown task, you should pick one in " + ",".join(task_to_keys.keys())
                )
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )


@dataclass
class FinetuneTrainingArguments(TrainingArguments):
    group_name: Optional[str] = field(default=None, metadata={"help": "W&B group name"})
    project_name: Optional[str] = field(default=None, metadata={"help": "Project name (W&B)"})
    early_stopping_patience: Optional[int] = field(
        default=-1, metadata={"help": "Early stopping patience value (default=-1 (disable))"}
    )
    # overriding to be True, for consistency with final_eval_{metric_name}
    fp16_full_eval: bool = field(
        default=True,
        metadata={"help": "Whether to use full 16-bit precision evaluation instead of 32-bit"},
    )



def convert_roberta_checkpoint_to_pytorch(
    roberta_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool, 
    checkpoint_name: str = "model.pt",
):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """
    SAMPLE_TEXT = "Hello world! cécé herlolip"

    roberta = FairseqRobertaModel.from_pretrained(roberta_checkpoint_path, checkpoint_file=checkpoint_name)
    roberta.eval()  # disable dropout
    roberta_sent_encoder = roberta.model.encoder.sentence_encoder
    roberta_mask_idx = roberta.model.encoder.dictionary.index("<mask>")
    
    config = RobertaConfig(
        vocab_size=roberta_sent_encoder.embed_tokens.num_embeddings,
        hidden_size=roberta.model.args.encoder_embed_dim, #roberta.args.encoder_embed_dim,
        num_hidden_layers=roberta.model.args.encoder_layers, #roberta.args.encoder_layers,
        num_attention_heads=roberta.model.args.encoder_attention_heads, #roberta.args.encoder_attention_heads,
        intermediate_size=roberta.model.args.encoder_ffn_embed_dim, #roberta.args.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
    )
    if classification_head:
        config.num_labels = roberta.model.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our BERT config:", config)

    model = RobertaForSequenceClassification(config) if classification_head else RobertaForMaskedLM(config)
    model.eval()

    hf_roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    hf_roberta_mask_idx = hf_roberta_tokenizer._convert_token_to_id("<mask>")
    # Now let's copy all the weights.
    # Embeddings
    model.roberta.embeddings.word_embeddings.weight = roberta_sent_encoder.embed_tokens.weight
    # bp()
    model.roberta.embeddings.word_embeddings.weight[hf_roberta_mask_idx].data = roberta_sent_encoder.embed_tokens.weight[roberta_mask_idx].data
    model.roberta.embeddings.position_embeddings.weight = roberta_sent_encoder.embed_positions.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c RoBERTa doesn't use them.
    
    # model.roberta.embeddings.LayerNorm.weight = roberta_sent_encoder.emb_layer_norm.weight
    # model.roberta.embeddings.LayerNorm.bias = roberta_sent_encoder.emb_layer_norm.bias
    # https://github.com/huggingface/transformers/issues/12670
    model.roberta.embeddings.LayerNorm.weight = roberta_sent_encoder.layernorm_embedding.weight
    model.roberta.embeddings.LayerNorm.bias = roberta_sent_encoder.layernorm_embedding.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer: BertLayer = model.roberta.encoder.layer[i]
        roberta_layer: TransformerSentenceEncoderLayer = roberta_sent_encoder.layers[i]

        # self attention
        self_attn: BertSelfAttention = layer.attention.self
        assert (
            roberta_layer.self_attn.k_proj.weight.data.shape
            == roberta_layer.self_attn.q_proj.weight.data.shape
            == roberta_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = roberta_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = roberta_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = roberta_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = roberta_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = roberta_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = roberta_layer.self_attn.v_proj.bias

        # self-attention output
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias
        self_output.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
        self_output.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias

        # intermediate
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
        intermediate.dense.weight = roberta_layer.fc1.weight
        intermediate.dense.bias = roberta_layer.fc1.bias

        # output
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
        bert_output.dense.weight = roberta_layer.fc2.weight
        bert_output.dense.bias = roberta_layer.fc2.bias
        bert_output.LayerNorm.weight = roberta_layer.final_layer_norm.weight
        bert_output.LayerNorm.bias = roberta_layer.final_layer_norm.bias
        # end of layer

    if classification_head:
        model.classifier.dense.weight = roberta.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = roberta.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = roberta.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = roberta.model.classification_heads["mnli"].out_proj.bias
    else:
        # LM Head
        model.lm_head.dense.weight = roberta.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = roberta.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = roberta.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = roberta.model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = roberta.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = roberta.model.encoder.lm_head.bias

    # Let's check that we get the same results.
    input_ids: torch.Tensor = roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1

    our_output = model(input_ids)[0]
    if classification_head:
        their_output = roberta.model.classification_heads["mnli"](roberta.extract_features(input_ids))
    else:
        their_output = roberta.model(input_ids)[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


def main(args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    unique_run_id = str(uuid.uuid1())

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FinetuneTrainingArguments))
    hyper_params = json.load(open(f'{args.finetune_config_dir}/{args.task}.json', 'r'))
    assert os.path.isfile(args.path_to_pretrained_checkpoint) and args.path_to_pretrained_checkpoint.endswith(".pt")
    pretrained_checkpoint_dir_path = os.path.dirname(args.path_to_pretrained_checkpoint)
    pretrained_checkpoint_name = os.path.basename(args.path_to_pretrained_checkpoint)
    dummy_link_file_name = f'{args.task}.{pretrained_checkpoint_name}'
    dummy_link_file_path = os.path.join(pretrained_checkpoint_dir_path, dummy_link_file_name)
    pytorch_dir_path = f"{pretrained_checkpoint_dir_path}/{pretrained_checkpoint_name.split('.')[0]}"
    os.makedirs(pytorch_dir_path, exist_ok=True)
    if args.overwrite_pytorch_dir:
        shutil.rmtree(pytorch_dir_path)
        os.remove(dummy_link_file_path)
        os.makedirs(pytorch_dir_path, exist_ok=True)
        
    hyper_params['model_name_or_path'] = pytorch_dir_path
    """
    convert models
    """
    if not os.path.exists(os.path.join(pytorch_dir_path, 'pytorch_model.bin')) or args.overwrite_pytorch_dir:
        # model_name = sorted([i for i in os.listdir(hyper_params['model_name_or_path']) if i.startswith('checkpoint')])[::-1][0]
        logger.info(f'converting fairseq model `{pretrained_checkpoint_name}` to pytorch model...')
        logger.info(f'Saving pytorch model to `{pytorch_dir_path}`...')
        if os.path.exists(dummy_link_file_path):
            os.remove(dummy_link_file_path)
        os.symlink(args.path_to_pretrained_checkpoint, dummy_link_file_path)
        # shutil.copy(os.path.join(hyper_params['model_name_or_path'], model_name), os.path.join(hyper_params['model_name_or_path'], 'model.pt'))
        convert_roberta_checkpoint_to_pytorch(
            roberta_checkpoint_path=pretrained_checkpoint_dir_path,
            pytorch_dump_folder_path=hyper_params['model_name_or_path'],
            classification_head=False,
            checkpoint_name=pretrained_checkpoint_name,
        )
    # bp()
    hyper_params['project_name'] = f'roberta_{args.masking}_{args.task}'
    if args.learning_rate is not None:
        hyper_params["learning_rate"] = args.learning_rate
    if args.per_device_train_batch_size is not None:
        hyper_params["per_device_train_batch_size"] = args.per_device_train_batch_size
    hyper_params['output_dir'] = f'{args.finetune_output_dir}/{args.masking}/{args.task}/lr{hyper_params["learning_rate"]}_B{hyper_params["per_device_train_batch_size"]}_E{hyper_params["num_train_epochs"]}'
    os.makedirs(hyper_params['output_dir'], exist_ok=True)
    with open(os.path.join(hyper_params['output_dir'], f'{args.task}.json'), 'w') as f:
        json.dump(hyper_params, f, indent=4)
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.join(hyper_params['output_dir'], f'{args.task}.json')
    )
    training_args.local_rank = args.local_rank

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.task_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset("glue", data_args.task_name)
    elif data_args.train_file.endswith(".csv"):
        # Loading a dataset from local csv files
        datasets = load_dataset(
            "csv",
            data_files={"train": data_args.train_file, "validation": data_args.validation_file},
        )
    else:
        # Loading a dataset from local json files
        datasets = load_dataset(
            "json",
            data_files={"train": data_args.train_file, "validation": data_args.validation_file},
        )
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Labels
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        if not is_regression:
            label_list = datasets["train"].features["label"].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
        if is_regression:
            num_labels = 1
        else:
            # A useful fast method:
            # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
            label_list = datasets["train"].unique("label")
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # pretrain_run_args = json.load(open(f"{model_args.model_name_or_path}/args.json", "r"))

    # def get_correct_ds_args(pretrain_run_args):
    #     ds_args = Namespace()
    #
    #     for k, v in pretrain_run_args.items():
    #         setattr(ds_args, k, v)
    #
    #     # to enable HF integration
    #     #         ds_args.huggingface = True
    #     return ds_args
    #
    # ds_args = get_correct_ds_args(pretrain_run_args)

    # # in so, deepspeed is required
    # if (
    #     "deepspeed_transformer_kernel" in pretrain_run_args
    #     and pretrain_run_args["deepspeed_transformer_kernel"]
    # ):
    #     logger.warning("Using deepspeed_config due to kernel usage")
    #
    #     remove_cuda_compatibility_for_kernel_compilation()

    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )
    # bp()
    config.hidden_dropout_prob = hyper_params['hidden_dropout_prob']
    config.attention_probs_dropout_prob = hyper_params['attention_probs_dropout_prob']

    tokenizer = AutoTokenizer.from_pretrained(
        'roberta-base',
        cache_dir=model_args.cache_dir,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        # args=ds_args,
    )

    # Preprocessing the datasets
    if data_args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
        non_label_column_names = [
            name for name in datasets["train"].column_names if name != "label"
        ]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
        max_length = data_args.max_seq_length
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False
        max_length = None

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if (
        model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        and data_args.task_name is not None
        and is_regression
    ):
        # Some have all caps in their config, some don't.
        label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
            label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        else:
            logger.warn(
                "Your model seems to have been trained with labels, but they don't match the dataset: ",
                f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
                "\nIgnoring the model labels as a result.",
            )
    elif data_args.task_name is None:
        label_to_id = {v: i for i, v in enumerate(label_list)}

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [label_to_id[l] for l in examples["label"]]
        return result

    datasets = datasets.map(
        preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache
    )

    train_dataset = datasets["train"]
    eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
    if data_args.task_name is not None:
        test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    try:
        import wandb

        run = wandb.init(
            project=training_args.project_name,
            group=training_args.group_name,
            name=training_args.run_name,
            dir="/tmp",
        )
        wandb.config.update(model_args)
        wandb.config.update(data_args)
        wandb.config.update(training_args)

    except Exception as e:
        logger.warning("W&B logger is not available, please install to get proper logging")
        logger.error(e)

    # init early stopping callback and metric to monitor
    callbacks = None
    if training_args.early_stopping_patience > 0:
        early_cb = EarlyStoppingCallback(training_args.early_stopping_patience)
        callbacks = [early_cb]

    # NOTE: modify this
    metric_monitor = {
        "mrpc": "f1",
        "sst2": "accuracy",
        "mnli": "accuracy",
        "mnli_mismatched": "accuracy",
        "mnli_matched": "accuracy",
        "cola": "matthews_correlation",
        "stsb": "spearmanr",
        "qqp": "f1",
        "qnli": "accuracy",
        "rte": "accuracy",
        "wnli": "accuracy",
    }
    metric_to_monitor = metric_monitor[data_args.task_name]
    setattr(training_args, "metric_for_best_model", metric_to_monitor)
    setattr(training_args, "load_best_model_at_end", True)
    setattr(training_args, "greater_is_better", True)
    setattr(training_args, 'save_total_limit', 1)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train()

    # Evaluation
    if training_args.do_eval:
        print("*** Evaluate ***")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        try:
            wandb.run.summary.update(metrics)
            log_metrics = {}
            for k, v in metrics.items():
                # log_metrics["final_" + k] = v
                log_metrics["best_" + k] = v
            wandb.log(log_metrics)
        except Exception as e:
            logger.warning("W&B logger is not available, please install to get proper logging")
            logger.error(e)

    if training_args.do_predict:
        logger.info("*** Test ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        test_datasets = [test_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            test_datasets.append(datasets["test_mismatched"])

        for test_dataset, task in zip(test_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            # test_dataset.remove_columns_("label")
            test_dataset = test_dataset.remove_columns("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            predictions = (
                np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)
            )

            test_results_file_name = f"test_results_{task}_{unique_run_id}.txt"
            # if os.path.isdir(model_args.model_name_or_path):
            #     output_test_file = os.path.join(
            #         model_args.model_name_or_path, test_results_file_name
            #     )
            # else:
            output_test_file = os.path.join(training_args.output_dir, test_results_file_name)

            print(f"test_results_file_name: {test_results_file_name}")

            if trainer.is_world_process_zero():
                with open(output_test_file, "w") as writer:
                    logger.info(f"***** Test results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")
    # save summary
    id = f"{run.entity}/{run.project}/{run.id}"

    return training_args.output_dir, id

def save_wandb_results(output_dir, id):
    import wandb
    summary = dict()
    api = wandb.Api()
    final_run = api.run(id)
    results = final_run.summary._json_dict
    for key, val in results.items():
        if type(val) not in [list, dict]:
            summary[key] = val
    with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)
    # download metrics
    metrics_dataframe = final_run.history()
    metrics_dataframe.to_csv(f"{output_dir}/metrics.csv")
    # remove model path
    # model_paths = [i for i in os.listdir(output_dir) if i.startswith('checkpoint') and os.path.isdir(os.path.join(training_args.output_dir, i))]
    # for one_model in model_paths:
    #     cur_path = os.path.join(output_dir, one_model)
    #     shutil.rmtree(cur_path)
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='reberta glue')
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--masking', type=str, required=True, choices=['mp0.15', 'mpr009_021', 'mp0.2', 'mp0.4', 'mp0.5', 'seq-len', 'seq-len_0_1_0_9', 'seq-len_0_2_0_8', 'seq-len_0_3_0_7'])
    parser.add_argument('--path_to_pretrained_checkpoint', type=str, required=True)
    parser.add_argument('--finetune_config_dir', type=str, required=True)
    parser.add_argument('--finetune_output_dir', type=str, required=True)
    parser.add_argument('--overwrite_pytorch_dir', action="store_true")
    parser.add_argument('--local_rank', type=int, default=-1)
    # changed for sweep
    parser.add_argument('--learning_rate', type=float, default=None)
    parser.add_argument('--per_device_train_batch_size', type=int, default=None)
    args = parser.parse_args()

    output_dir, id = main(args)
    save_wandb_results(output_dir, id)
