# Created by xunannancy at 2022/5/1
# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import json
import logging
import os
import random
import shutil
import sys
from argparse import Namespace
# from pretraining.args.deepspeed_args import remove_cuda_compatibility_for_kernel_compilation
# from pretraining.modeling import BertForSequenceClassification
# from pretraining.configs import PretrainedBertConfig
from dataclasses import dataclass, field
from typing import Optional
import uuid

import numpy as np
import transformers
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    PreTrainedTokenizerFast
)
from transformers.trainer_utils import SchedulerType, is_main_process

"""
A subclass of `Trainer` specific to Question-Answering tasks
"""

from transformers import Trainer, is_torch_tpu_available
from transformers.trainer_utils import PredictionOutput
from typing import Optional, Tuple
import collections
from tqdm.auto import tqdm
import datasets
from roberta_run_glue import convert_roberta_checkpoint_to_pytorch
import argparse

logger = logging.getLogger(__name__)

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        # We might have removed columns from the dataset so we put them back.
        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        eval_preds = self.post_process_function(test_examples, test_dataset, output.predictions)
        metrics = self.compute_metrics(eval_preds)

        return PredictionOutput(predictions=eval_preds.predictions, label_ids=eval_preds.label_ids, metrics=metrics)

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this process is the main process or not (used to determine if logging/saves should be done).
    """
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.
                    # NOTE: from transformers v4.4.0
                    # if (
                    #     start_index >= len(offset_mapping)
                    #     or end_index >= len(offset_mapping)
                    #     or offset_mapping[start_index] is None
                    #     or offset_mapping[end_index] is None
                    # ):
                    #     continue
                    # NOTE: from latest transformer
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or len(offset_mapping[start_index]) < 2
                        or offset_mapping[end_index] is None
                        or len(offset_mapping[end_index]) < 2
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # # If we have an output_dir, let's save all those dicts.
    # if output_dir is not None:
    #     assert os.path.isdir(output_dir), f"{output_dir} is not a directory."
    #
    #     prediction_file = os.path.join(
    #         output_dir, "predictions.json" if prefix is None else f"predictions_{prefix}".json
    #     )
    #     nbest_file = os.path.join(
    #         output_dir, "nbest_predictions.json" if prefix is None else f"nbest_predictions_{prefix}".json
    #     )
    #     if version_2_with_negative:
    #         null_odds_file = os.path.join(
    #             output_dir, "null_odds.json" if prefix is None else f"null_odds_{prefix}".json
    #         )
    #
    #     logger.info(f"Saving predictions to {prediction_file}.")
    #     with open(prediction_file, "w") as writer:
    #         writer.write(json.dumps(all_predictions, indent=4) + "\n")
    #     logger.info(f"Saving nbest_preds to {nbest_file}.")
    #     with open(nbest_file, "w") as writer:
    #         writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
    #     if version_2_with_negative:
    #         logger.info(f"Saving null_odds to {null_odds_file}.")
    #         with open(null_odds_file, "w") as writer:
    #             writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
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
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
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


def main(args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, FinetuneTrainingArguments))
    hyper_params = json.load(open(f'{args.finetune_config_dir}/{args.task}.json', 'r'))
    pretrained_checkpoint_dir_path = os.path.dirname(args.path_to_pretrained_checkpoint)
    pretrained_checkpoint_name = os.path.basename(args.path_to_pretrained_checkpoint)
    pytorch_dir_path = f"{pretrained_checkpoint_dir_path}/{pretrained_checkpoint_name.split('.')[0]}"
    os.makedirs(pytorch_dir_path, exist_ok=True)
    if args.overwrite_pytorch_dir:
        shutil.rmtree(pytorch_dir_path)
        os.remove(os.path.join(pretrained_checkpoint_dir_path, 'model.pt'))
        os.makedirs(pytorch_dir_path, exist_ok=True)
    
    hyper_params['model_name_or_path'] = pytorch_dir_path
    """
    convert models
    """
    if not os.path.exists(os.path.join(pytorch_dir_path, 'pytorch_model.bin')) or args.overwrite_pytorch_dir:
        # model_name = sorted([i for i in os.listdir(hyper_params['model_name_or_path']) if i.startswith('checkpoint')])[::-1][0]
        logger.info(f'converting fairseq model `{pretrained_checkpoint_name}` to pytorch model...')
        logger.info(f'Saving pytorch model to `{pytorch_dir_path}`...')
        dummy_file = os.path.join(pretrained_checkpoint_dir_path, 'model.pt')
        if os.path.exists(dummy_file):
            os.remove(dummy_file)
        os.symlink(args.path_to_pretrained_checkpoint, dummy_file)
        # shutil.copy(os.path.join(hyper_params['model_name_or_path'], model_name), os.path.join(hyper_params['model_name_or_path'], 'model.pt'))
        convert_roberta_checkpoint_to_pytorch(
            roberta_checkpoint_path=pretrained_checkpoint_dir_path,
            pytorch_dump_folder_path=hyper_params['model_name_or_path'],
            classification_head=False
        )
    hyper_params['project_name'] = f'roberta_{args.masking}_{args.task}'
    hyper_params['output_dir'] = f'{args.finetune_output_dir}/{args.masking}/{args.task}/lr{hyper_params["learning_rate"]}_B{hyper_params["per_device_train_batch_size"]}_E{hyper_params["num_train_epochs"]}'
    os.makedirs(hyper_params['output_dir'], exist_ok=True)
    with open(os.path.join(hyper_params['output_dir'], f'{args.task}.json'), 'w') as f:
        json.dump(hyper_params, f, indent=4)
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.join(hyper_params['output_dir'], f'{args.task}.json')
    )

    # If we pass only one argument to the script and it's the path to a json file,
    # let's parse it to get our arguments.
    model_args, data_args, training_args = parser.parse_json_file(
        json_file=os.path.join(hyper_params['output_dir'], f'{args.task}.json')
    )

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
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name)
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]

        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]

        datasets = load_dataset(extension, data_files=data_files, field="data")

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    # pretrain_run_args = json.load(open(f"{model_args.model_name_or_path}/args.json", "r"))
    #
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
    #
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
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.hidden_dropout_prob = hyper_params['hidden_dropout_prob']
    config.attention_probs_dropout_prob = hyper_params['attention_probs_dropout_prob']

    tokenizer = AutoTokenizer.from_pretrained(
        'roberta-base',
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#supported-frameworks to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the datasets.
    # Preprocessing is slighlty different for training and evaluation.
    if training_args.do_train:
        column_names = datasets["train"].column_names
    else:
        column_names = datasets["validation"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    print(f'data_args.max_seq_length: {data_args.max_seq_length}, tokenizer.model_max_length: {tokenizer.model_max_length} => max_seq_length: {max_seq_length}')

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        """
        NOTE: problem of squad for roberta: too many spaces: (old dataset version: 2.0.0)
        https://github.com/huggingface/transformers/issues/12880
        https://github.com/huggingface/transformers/commit/3ec851dc5ed303961574b781d6c794551b68edc0#diff-85cadb0c6b23d2fa980473fba25c11d9a7ce27aedbc957697be6a40d2305f568
        """
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            # We will select sample from whole data if agument is specified
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        # Create train feature from dataset
        train_dataset = train_dataset.map(
            prepare_train_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.max_train_samples is not None:
            # Number of samples might increase during Feature Creation, We select only specified max samples
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    # Validation preprocessing
    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples[question_column_name] = [q.lstrip() for q in examples[question_column_name]]

        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    if training_args.do_eval:
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        if data_args.max_val_samples is not None:
            # We will select sample from whole data
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        # Validation Feature Creation
        eval_dataset = eval_dataset.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        if data_args.max_val_samples is not None:
            # During Feature creation dataset samples might increase, we will select required samples again
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))



    # Data collator
    # We have already padded to max length if the corresponding flag is True, otherwise we need to pad in the data
    # collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Post-processing:
    def post_processing_function(examples, features, predictions):
        # Post-processing: we match the start logits and end logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            is_world_process_zero=trainer.is_world_process_zero(),
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]
        references = [{"id": ex["id"], "answers": ex[answer_column_name]} for ex in datasets["validation"]]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

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

    setattr(training_args, "metric_for_best_model", 'f1')
    setattr(training_args, "load_best_model_at_end", True)
    setattr(training_args, "greater_is_better", True)
    setattr(training_args, 'save_total_limit', 1)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=datasets["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        callbacks=callbacks,
        data_collator=data_collator,
        post_process_function=post_processing_function,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))
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
    parser.add_argument('-task', type=str, required=True)
    parser.add_argument('-masking', type=str, required=True, choices=['mp0.15', 'mp0.4', 'mp0.5', 'seq-len', 'seq-len_0_1_0_9', 'seq-len_0_2_0_8', 'seq-len_0_3_0_7'])
    parser.add_argument('-path_to_pretrained_checkpoint', type=str, required=True)
    parser.add_argument('-finetune_config_dir', type=str, required=True)
    parser.add_argument('-finetune_output_dir', type=str, required=True)
    parser.add_argument('-overwrite_pytorch_dir', action="store_true")
    args = parser.parse_args()

    output_dir, id = main(args)
    save_wandb_results(output_dir, id)
