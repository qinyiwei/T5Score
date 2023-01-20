
import torch
import logging
import os
import sys
import glob
import json
from dataclasses import dataclass, field
from typing import Optional

import transformers
from discriminative_training.supervised_model import EstimatorT5
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    MBartTokenizer,
    # Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
    EncoderDecoderModel,
)
from discriminative_training.trainer import SupMultiEvalTrainer, EstmMultiEvalTrainer, BrioTrainer

from transformers.trainer_utils import is_main_process
from transformers.training_args import ParallelMode
from discriminative_training.utils import (
    Seq2SeqDataCollator,
    Seq2SeqDataset,
    MultiDataset,
    assert_all_frozen,
    build_compute_metrics_fn,
    check_output_dir,
    freeze_embeds,
    freeze_params,
    lmap,
    save_json,
    use_task_specific_params,
    write_txt_file,
)

from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler


logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    freeze_encoder: bool = field(default=False, metadata={
                                 "help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={
                                "help": "Whether  to freeze the embeddings."})

    tie_encoder_decoder: bool = field(
        default=False,
        metadata={"help": "tie encoder decoder"},
    )

    use_estimator: bool = field(
        default=False,
        metadata={"help": "use estimator model"},
    )

    no_layer_norm: bool = field(
        default=False,
        metadata={"help": "use layer norm"},
    )

    dropout_rate: float = field(
        default=0.1,
        metadata={"help": "dropout rate"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    data_dir: str = field(
        metadata={
            "help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    task: Optional[str] = field(
        default="summarization",
        metadata={
            "help": "Task name, summarization (or summarization_{dataset} for pegasus) or translation"},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=84,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. "
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=84,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(
        default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(
        default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(
        default=-1, metadata={"help": "# test examples. -1 means use all."})
    src_lang: Optional[str] = field(
        default=None, metadata={"help": "Source language id for translation."})
    tgt_lang: Optional[str] = field(
        default=None, metadata={"help": "Target language id for translation."})
    eval_beams: Optional[int] = field(
        default=4, metadata={"help": "# num_beams to use for evaluation."})
    length_penalty: Optional[float] = field(
        default=0.6, metadata={"help": "# length_penalty"})
    no_repeat_ngram_size: Optional[int] = field(
        default=None, metadata={"help": "# num_beams to use for evaluation."})
    upsampling_factor: Optional[float] = field(default=None,
                                               metadata={
                                                   "help": "# use data upsampling factor only when using multiple data files."}
                                               )
    rouge_lang: Optional[str] = field(default=None,
                                      metadata={
                                          "help": "# apply language specific tokenization and stemming (if available)"}
                                      )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )

    references: Optional[str] = field(
        default="ref",
        metadata={"help": "could be ref|src|ref_src"},
    )

    direction: Optional[str] = field(
        default="single_direction",
        metadata={
            "help": "could be single_direction|double_direction|combine_direction"},
    )
    use_standardlize: bool = field(
        default=False,
        metadata={"help": "use standardlize or not."},
    )

    brio_loss: bool = field(
        default=False,
        metadata={"help": "use brio loss"},
    )

    margin: Optional[float] = field(
        default=1, metadata={"help": "brio loss margin"}
    )

    loss_type: str = field(
        default="margin",
        metadata={
            "help": "could be margin|hybrid_H|hybrid_L|hybrid_B|hybrid_H_filter"},
    )

    loss_beta: float = field(
        default=1,
        metadata={"help": "weight of margin loss in hybrid loss type"},
    )

    loss_alpha: float = field(
        default=1,
        metadata={"help": "weight of nll loss in hybrid loss type"},
    )

    loss_th: float = field(
        default=0,
        metadata={"help": "manual score threshold to calculate nll"},
    )

    model_parallel: bool = field(
        default=False,
        metadata={
            "help": (
                "If there are more than one devices, whether to use model parallelism to distribute the "
                "model's modules across devices."
            )
        },
    )


def handle_metrics(split, metrics, output_dir):
    """
    Log and save metrics

    Args:
    - split: one of train, val, test
    - metrics: metrics dict
    - output_dir: where to save the metrics
    """

    logger.info(f"***** {split} metrics *****")
    for key in sorted(metrics.keys()):
        logger.info(f"  {key} = {metrics[key]}")
    save_json(metrics, os.path.join(output_dir, f"{split}_results.json"))


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.direction = data_args.direction
    training_args.brio_loss = data_args.brio_loss
    training_args.margin = data_args.margin
    training_args.loss_type = data_args.loss_type
    training_args.loss_beta = data_args.loss_beta
    training_args.loss_alpha = data_args.loss_alpha
    training_args.loss_th = data_args.loss_th
    training_args.model_parallel = data_args.model_parallel

    check_output_dir(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [
            -1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.parallel_mode == ParallelMode.DISTRIBUTED),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {}
    if data_args.max_target_length:
        config_kwargs.update({'max_length': data_args.max_target_length})
    if data_args.eval_beams:
        config_kwargs.update({'num_beams': data_args.eval_beams})
    if data_args.length_penalty:
        config_kwargs.update({'length_penalty': data_args.length_penalty})
    if data_args.no_repeat_ngram_size:
        config_kwargs.update(
            {'no_repeat_ngram_size': data_args.no_repeat_ngram_size})

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir, **config_kwargs,
    )
    setattr(config, 'dropout_rate', model_args.dropout_rate)
    print(config)

    extra_model_params = ("encoder_layerdrop",
                          "decoder_layerdrop", "dropout", "attention_dropout")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(
                config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        use_fast=False, cache_dir=model_args.cache_dir,
    )

    if not model_args.use_estimator:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    else:
        model = EstimatorT5.from_pretrained(
            model_args.model_name_or_path,
            from_tf=".ckpt" in model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            layer_norm=not model_args.no_layer_norm,
        )
    # use task specific params
    use_task_specific_params(model, data_args.task)

    if training_args.model_parallel:
        model.parallelize()

    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # set decoder_start_token_id for MBart
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, MBartTokenizer):
        assert (
            data_args.tgt_lang is not None and data_args.src_lang is not None
        ), "mBart requires --tgt_lang and --src_lang"
        model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.tgt_lang]

    if model_args.freeze_embeds:
        freeze_embeds(model)
    if model_args.freeze_encoder:
        freeze_params(model.get_encoder())
        assert_all_frozen(model.get_encoder())

    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if training_args.local_rank != -1 else 1)
    )
    logger.info(f'Effective batch size: {total_train_batch_size}')

    dataset_class = Seq2SeqDataset
    dataset_collator = Seq2SeqDataCollator(
        tokenizer, data_args, None, training_args.tpu_num_cores, use_estimator=model_args.use_estimator)

    assert training_args.do_train, "only surpport training"
    # Get datasets
    if data_args.upsampling_factor is not None:
        multi_dataset_kwargs = {
            "upsampling_factor": data_args.upsampling_factor,
            "total_batch_size": total_train_batch_size,
            "actual_batch_size": training_args.train_batch_size,
            "gradient_accum": training_args.gradient_accumulation_steps,
            "is_distributed": bool(training_args.local_rank != -1),
            "dataset_class": dataset_class
        }
        train_dataset = (
            MultiDataset(
                tokenizer,
                type_path="train",
                data_dir=data_args.data_dir,
                n_obs=data_args.n_train,
                max_target_length=data_args.max_target_length,
                max_source_length=data_args.max_source_length,
                prefix=model.config.prefix or "",
                references=data_args.references,
                direction=data_args.direction,
                use_standardlize=data_args.use_standardlize,
                brio_loss=data_args.brio_loss,
                **multi_dataset_kwargs,
            )
            if training_args.do_train
            else None
        )
    else:
        train_dataset = (
            dataset_class(
                tokenizer,
                type_path="train",
                data_dir=data_args.data_dir,
                n_obs=data_args.n_train,
                max_target_length=data_args.max_target_length,
                max_source_length=data_args.max_source_length,
                prefix=model.config.prefix or "",
                references=data_args.references,
                direction=data_args.direction,
                use_standardlize=data_args.use_standardlize,
                brio_loss=data_args.brio_loss,
            )
            if training_args.do_train
            else None
        )

    # Initialize our Trainer
    compute_metrics_fn = (
        build_compute_metrics_fn(
            data_args.task, tokenizer, data_args) if training_args.predict_with_generate else None
    )
    if model_args.use_estimator:
        trainer = EstmMultiEvalTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=dataset_collator,
            compute_metrics=compute_metrics_fn,
            tokenizer=tokenizer,
        )
    else:
        if data_args.brio_loss:
            trainer = BrioTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=dataset_collator,
                compute_metrics=compute_metrics_fn,
                tokenizer=tokenizer,
            )
        else:
            trainer = SupMultiEvalTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator=dataset_collator,
                compute_metrics=compute_metrics_fn,
                tokenizer=tokenizer,
            )

    all_metrics = {}
    # Training
    if training_args.do_train:
        logger.info("*** Train ***")

        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(
                model_args.model_name_or_path) else None
        )
        metrics = train_result.metrics
        metrics["train_n_objs"] = data_args.n_train

        trainer.save_model()  # this also saves the tokenizer

        if trainer.is_world_process_zero():
            handle_metrics("train", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(
                training_args.output_dir, "trainer_state.json"))

            # For convenience, we also re-save the tokenizer to the same directory,
            # so that you can share your model easily on huggingface.co/models =)
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(
            metric_key_prefix="val", max_length=data_args.val_max_target_length, num_beams=data_args.eval_beams
        )
        metrics["val_n_objs"] = data_args.n_val
        metrics["val_loss"] = round(metrics["val_loss"], 4)

        if trainer.is_world_process_zero():

            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        test_output = trainer.predict(
            test_dataset=test_dataset,
            metric_key_prefix="test",
            max_length=data_args.val_max_target_length,
            num_beams=data_args.eval_beams,
            length_penalty=data_args.length_penalty,
            no_repeat_ngram_size=data_args.no_repeat_ngram_size,
        )
        metrics = test_output.metrics
        metrics["test_n_objs"] = data_args.n_test

        if trainer.is_world_process_zero():
            metrics["test_loss"] = round(metrics["test_loss"], 4)
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

            if training_args.predict_with_generate:
                predictions = test_output.predictions
                predictions[predictions == -100] = tokenizer.pad_token_id
                test_preds = tokenizer.batch_decode(
                    predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                test_preds = lmap(str.strip, test_preds)
                write_txt_file(test_preds, os.path.join(
                    training_args.output_dir, "test_generations.txt"))

    if trainer.is_world_process_zero():
        save_json(all_metrics, os.path.join(
            training_args.output_dir, "all_results.json"))

    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
