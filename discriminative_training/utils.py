import itertools
import json
import linecache
import math
import os
import pickle
import socket
import glob
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import csv
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, Sampler, RandomSampler, DataLoader

from transformers import BartTokenizer, EvalPrediction, PreTrainedTokenizer, T5Tokenizer, BertTokenizer, RobertaTokenizer
from transformers.file_utils import cached_property
from transformers.models.bart.modeling_bart import shift_tokens_right

import copy

try:
    from fairseq.data.data_utils import batch_by_size

    FAIRSEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRSEQ_AVAILABLE = False



def lmap(f: Callable, x: Iterable) -> List:
    """list(map(f, x))"""
    return list(map(f, x))


def build_compute_metrics_fn(task_name: str, tokenizer: PreTrainedTokenizer, data_args) -> Callable[[EvalPrediction], Dict]:
    def non_pad_len(tokens: np.ndarray) -> int:
        return np.count_nonzero(tokens != tokenizer.pad_token_id)

    def decode_pred(pred: EvalPrediction) -> Tuple[List[str], List[str]]:
        predictions = pred.predictions
        label_ids = pred.label_ids
        predictions[predictions == -100] = tokenizer.pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pred_str = lmap(str.strip, pred_str)
        label_str = lmap(str.strip, label_str)
        return pred_str, label_str

    def summarization_metrics(pred: EvalPrediction) -> Dict:
        #pred_str, label_str = decode_pred(pred)
        #rouge: Dict = calculate_rouge(
        #    pred_str, label_str,
        #    rouge_lang=data_args.rouge_lang,
        #)
        #summ_len = np.round(np.mean(lmap(non_pad_len, pred.predictions)), 1)
        #rouge.update({"gen_len": summ_len})
        rouge = 0
        return rouge


    compute_metrics_fn = summarization_metrics 
    return compute_metrics_fn


def trim_batch(
    input_ids,
    pad_token_id,
    attention_mask=None,
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])



class MultiDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        n_obs=None,
        prefix="",
        is_bart=False,
        **dataset_kwargs
    ):
        super().__init__()
        assert "upsampling_factor" in dataset_kwargs, "upsampling_factor required"
        assert "total_batch_size" in dataset_kwargs, "total_batch_size required"
        assert "actual_batch_size" in dataset_kwargs, "actual_batch_size required"
        assert "gradient_accum" in dataset_kwargs, "gradient_accum required"
        assert "is_distributed" in dataset_kwargs, "is_distributed required"
        assert "dataset_class" in dataset_kwargs, "dataset_class required"
        
        self.dataloaders = []
        self.total_batch_size = dataset_kwargs.pop("total_batch_size")
        dataset_class = dataset_kwargs.pop("dataset_class")
        references = dataset_kwargs.pop("references") 
        direction = dataset_kwargs.pop("direction")
        use_standardlize = dataset_kwargs.pop("use_standardlize")
        brio_loss = dataset_kwargs.pop("brio_loss")
        extension = "csv" 
        # identify all source training files
        datasets = []
        self.source_files = glob.glob(os.path.join(data_dir, f'*.{extension}'))
        self.source_files.sort()
        for src_file in self.source_files:
            dataset = dataset_class(
                tokenizer,
                data_dir=src_file,
                n_obs=n_obs,
                max_target_length=max_target_length,
                max_source_length=max_source_length,
                prefix=prefix,
                references = references, 
                direction = direction,
                use_standardlize = use_standardlize,
                brio_loss = brio_loss,
            )
            datasets.append(dataset)
            train_sampler = RandomSampler(dataset)
            dataloader = DataLoader(
                dataset,
                batch_size=1,
                sampler=train_sampler,
                collate_fn=lambda batch: batch
            )
            self.dataloaders.append(dataloader)
        assert len(self.dataloaders) > 1, "multiple source/target filepairs required for MultiDataset"
        # compute effective length of this dataset and the sampling probabilities
        logger.info(f"Found datasets: {len(self.dataloaders)}")
        upsampling_factor = dataset_kwargs.get("upsampling_factor")
        datapoint_counts = np.array([len(dataset) for dataset in datasets])
        logger.info(f"Total datapoints: {np.sum(datapoint_counts)}")
        
        datapoint_probs = datapoint_counts / datapoint_counts.sum()
        smoothed_probs = datapoint_probs ** upsampling_factor

        self.sampling_probs = smoothed_probs / smoothed_probs.sum()
        self.effective_length = int(np.sum(datapoint_counts * self.sampling_probs))
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]

        is_distributed = dataset_kwargs.get("is_distributed")
        actual_batch_size = dataset_kwargs.get("actual_batch_size")
        gradient_accum = dataset_kwargs.get("gradient_accum")
        self.per_gpu_effective_batch_size = actual_batch_size * gradient_accum
            
        rank = int(os.environ.get("RANK")) if is_distributed else -1
        self.pos_shift_count = rank * self.per_gpu_effective_batch_size
        logger.info(f'Rank: {rank}, shifting required: {self.pos_shift_count}')
                
        self.current_dataset_idx = -1
        self.current_loader_count = 0
        self.is_bart = is_bart

    
    def shift_iterator(self, idx, shift_count):
        if shift_count <= 0:
            return
        iterator = self.iterators[idx]
        for _ in range(shift_count):
            try:
                next(iterator)
            except StopIteration:
                dataloader = self.dataloaders[idx]
                iterator = iter(dataloader)
                
        self.iterators[idx] = iterator

    def __len__(self):
        return self.effective_length

    def __getitem__(self, index):
        if self.current_loader_count == 0:
            self.current_dataset_idx = np.random.choice(range(len(self.dataloaders)), p=self.sampling_probs)
            # start of a new effective batch, shift to appropriate pos
            self.shift_iterator(self.current_dataset_idx, self.pos_shift_count)
            
        iterator = self.iterators[self.current_dataset_idx]
        self.current_loader_count = (self.current_loader_count + 1) % self.total_batch_size
        
        try:
            datapoint = next(iterator)
        except StopIteration:
            dataloader = self.dataloaders[self.current_dataset_idx]
            self.iterators[self.current_dataset_idx] = iter(dataloader)
            datapoint = next(self.iterators[self.current_dataset_idx])

        if self.current_loader_count == self.per_gpu_effective_batch_size:
            # taken allocated datapoints from this effective batch, move to the start of next batch
            self.shift_iterator(self.current_dataset_idx, self.total_batch_size - self.current_loader_count - self.pos_shift_count)
            self.current_loader_count = 0

        if self.is_bart:
            datapoint[0]['lang_id'] = self.current_dataset_idx

        return datapoint[0]



        
class AbstractSeq2SeqDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        data_dir,
        max_source_length,
        max_target_length,
        n_obs=None,
        prefix="",
        **dataset_kwargs
    ):
        super().__init__()
        self.data_file = open(data_dir)
        csvreader = csv.reader(self.data_file,delimiter=' ')
        header = next(csvreader)
        self.dataset_kwargs = dataset_kwargs
        
        self.srcs = [] 
        if self.dataset_kwargs['references'] == "ref":
            self.refs = []

        if dataset_kwargs['brio_loss']:
            self.srcs_L = [] 
            self.syss_H = []
            self.syss_L = []
            self.manualZ_H = []
            self.manualZ_L = []
            for row in csvreader:
                if self.dataset_kwargs['references'] == "ref" and (row[header.index('manualZ1')] == '' \
                    or row[header.index('src1')] == '' or row[header.index('ref1')] == '' or row[header.index('sys1')] == '' \
                    or row[header.index('manualZ2')] == '' \
                    or row[header.index('src2')] == '' or row[header.index('ref2')] == '' or row[header.index('sys2')] == '' ):
                    continue
                if self.dataset_kwargs['references'] == "src" and (row[header.index('manualZ1')] == '' \
                    or row[header.index('src1')] == '' or row[header.index('sys1')] == '' \
                    or row[header.index('manualZ2')] == '' \
                    or row[header.index('src2')] == '' or row[header.index('sys2')] == '' ):
                    continue
                if self.dataset_kwargs['references'] == "ref":
                    self.refs.append(row[header.index('ref1')])
                srcs1 = row[header.index('src1')]
                srcs2 = row[header.index('src2')]
                syss1 = row[header.index('sys1')]
                syss2 = row[header.index('sys2')]
                manualZ1 = float(row[header.index('manualZ1')])
                manualZ2 = float(row[header.index('manualZ2')])
                if manualZ1 > manualZ2:
                    self.srcs.append(srcs1)
                    self.srcs_L.append(srcs2)
                    self.syss_H.append(syss1)
                    self.manualZ_H.append(manualZ1)   
                    self.syss_L.append(syss2)
                    self.manualZ_L.append(manualZ2) 
                else:
                    self.srcs.append(srcs2)
                    self.srcs_L.append(srcs1)
                    self.syss_H.append(syss2)
                    self.manualZ_H.append(manualZ2)   
                    self.syss_L.append(syss1)
                    self.manualZ_L.append(manualZ1) 

            self.src_lens = self.get_char_lens(self.srcs)
            self.src_L_lens = self.get_char_lens(self.srcs_L)
            if self.dataset_kwargs['references'] == "ref":
                self.ref_lens = self.get_char_lens(self.refs)
            self.sys_H_lens = self.get_char_lens(self.syss_H)
            self.sys_L_lens = self.get_char_lens(self.syss_L)

            self.used_char_len = True
            
            self.max_source_length = max_source_length
            self.max_target_length = max_target_length
            assert min(self.src_lens) > 0, f"found empty src line in {self.data_dir}"
            assert min(self.src_L_lens) > 0, f"found empty src_L line in {self.data_dir}"
            if self.dataset_kwargs['references'] == "ref":
                assert min(self.ref_lens) > 0, f"found empty ref line in {self.data_dir}"
            assert min(self.sys_H_lens) > 0, f"found empty sys_H line in {self.data_dir}"
            assert min(self.sys_L_lens) > 0, f"found empty sys_L line in {self.data_dir}"


            assert self.dataset_kwargs['direction'] == 'single_direction' \
                or self.dataset_kwargs['direction'] == 'double_direction'
            assert self.dataset_kwargs['references'] == "ref" \
                or self.dataset_kwargs['references'] == "src"

            if self.dataset_kwargs['direction'] == 'single_direction':
                if self.dataset_kwargs['references'] == "ref":
                    self.source_lines = self.refs
                    self.source_lines_L = self.refs
                if self.dataset_kwargs['references'] == "src":
                    self.source_lines = self.srcs
                    self.source_lines_L = self.srcs_L
                self.tgt_lines_H = self.syss_H
                self.tgt_lines_L = self.syss_L
            if self.dataset_kwargs['direction'] == 'double_direction':
                if self.dataset_kwargs['references'] == "ref":
                    self.source_lines = copy.deepcopy(self.refs)
                    self.source_lines_L = copy.deepcopy(self.refs)
                if self.dataset_kwargs['references'] == "src":
                    self.source_lines = copy.deepcopy(self.srcs)
                    self.source_lines_L = copy.deepcopy(self.srcs_L)
                self.tgt_lines_H = copy.deepcopy(self.syss_H)
                self.tgt_lines_L = copy.deepcopy(self.syss_L)

                self.source_lines.extend(self.syss_H)
                self.source_lines_L.extend(self.syss_L)
                if self.dataset_kwargs['references'] == "ref":
                    self.tgt_lines_H.extend(self.refs)
                    self.tgt_lines_L.extend(self.refs)
                if self.dataset_kwargs['references'] == "src":
                    self.tgt_lines_H.extend(self.srcs)
                    self.tgt_lines_L.extend(self.srcs_L)

                self.manualZ_H.extend(self.manualZ_H)
                self.manualZ_L.extend(self.manualZ_L)

            if self.dataset_kwargs['use_standardlize'] == True:
                self.manualZ_H = self.standardlize(self.manualZ_H)
                self.manualZ_L = self.standardlize(self.manualZ_L)

            print("dataset source_H number:{}".format(len(self.source_lines)))
            print("dataset source_L number:{}".format(len(self.source_lines_L)))
            print("dataset target_H number:{}".format(len(self.tgt_lines_H)))
            print("dataset target_L number:{}".format(len(self.tgt_lines_L)))
            print("dataset manualZ_H scores number:{}".format(len(self.manualZ_H)))
            print("dataset manualZ_L scores number:{}".format(len(self.manualZ_L)))


            self.tokenizer = tokenizer
            self.prefix = prefix if prefix is not None else ""

            if n_obs is not None and n_obs != -1:
                self.source_lines = self.source_lines[:n_obs]
                self.source_lines_L = self.source_lines_L[:n_obs]
            self.pad_token_id = self.tokenizer.pad_token_id
            self.dataset_kwargs = dataset_kwargs
            dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

        else:
            self.syss = []
            self.manualZ = []
            for row in csvreader:
                if row[header.index('manualZ')] == '' \
                    or row[header.index('src')] == '' or row[header.index('ref')] == '' or row[header.index('sys')] == '':
                    continue
                self.srcs.append(row[header.index('src')])
                self.refs.append(row[header.index('ref')])
                self.syss.append(row[header.index('sys')])
                self.manualZ.append(float(row[header.index('manualZ')]))           

            self.src_lens = self.get_char_lens(self.srcs)
            self.ref_lens = self.get_char_lens(self.refs)
            self.sys_lens = self.get_char_lens(self.syss)

            self.used_char_len = True
            
            self.max_source_length = max_source_length
            self.max_target_length = max_target_length
            assert min(self.src_lens) > 0, f"found empty src line in {self.data_dir}"
            assert min(self.ref_lens) > 0, f"found empty ref line in {self.data_dir}"
            assert min(self.sys_lens) > 0, f"found empty sys line in {self.data_dir}"

            if self.dataset_kwargs['references'] == "ref":
                if self.dataset_kwargs['direction'] == 'single_direction' or self.dataset_kwargs['direction'] == 'combine_direction':
                    self.source_lines = self.refs
                    self.tgt_lines = self.syss
                elif self.dataset_kwargs['direction'] == 'double_direction':
                    self.source_lines = copy.deepcopy(self.refs)
                    self.tgt_lines = copy.deepcopy(self.syss)
                    self.source_lines.extend(self.syss)
                    self.tgt_lines.extend(self.refs)
                    self.manualZ.extend(self.manualZ)
                else:
                    raise NotImplementedError("not support {} as direction".format(self.dataset_kwargs['direction']))

            elif self.dataset_kwargs['references'] == "src":
                if self.dataset_kwargs['direction'] == 'single_direction' or self.dataset_kwargs['direction'] == 'combine_direction':
                    self.source_lines = self.srcs
                    self.tgt_lines = self.syss
                elif self.dataset_kwargs['direction'] == 'double_direction':
                    self.source_lines = copy.deepcopy(self.srcs)
                    self.tgt_lines = copy.deepcopy(self.syss)
                    self.source_lines.extend(self.syss)
                    self.tgt_lines.extend(self.srcs)
                    self.manualZ.extend(self.manualZ)
                else:
                    raise NotImplementedError("not support {} as direction".format(self.dataset_kwargs['direction']))

            elif self.dataset_kwargs['references'] == "ref_src":
                if self.dataset_kwargs['direction'] == 'single_direction' or self.dataset_kwargs['direction'] == 'combine_direction':
                    self.source_lines = copy.deepcopy(self.refs)
                    self.tgt_lines = copy.deepcopy(self.syss)
                    self.source_lines.extend(self.srcs)
                    self.tgt_lines.extend(self.syss)
                    self.manualZ.extend(self.manualZ)
                elif self.dataset_kwargs['direction'] == 'double_direction':
                    self.source_lines = copy.deepcopy(self.refs)
                    self.tgt_lines = copy.deepcopy(self.syss)
                    self.source_lines.extend(self.srcs)
                    self.source_lines.extend(self.syss)
                    self.source_lines.extend(self.syss)
                    self.tgt_lines.extend(self.syss)
                    self.tgt_lines.extend(self.refs)
                    self.tgt_lines.extend(self.srcs)
                    self.manualZ.extend(self.manualZ)
                    self.manualZ.extend(self.manualZ)
                else:
                    raise NotImplementedError("not support {} as direction".format(self.dataset_kwargs['direction']))

            else: 
                raise NotImplementedError("not support {} as references".format(self.dataset_kwargs['references']))

            if self.dataset_kwargs['use_standardlize'] == True:
                self.manualZ = self.standardlize(self.manualZ)

            print("dataset source number:{}".format(len(self.source_lines)))
            print("dataset target number:{}".format(len(self.tgt_lines)))
            print("dataset manual scores number:{}".format(len(self.manualZ)))


            self.tokenizer = tokenizer
            self.prefix = prefix if prefix is not None else ""

            if n_obs is not None and n_obs != -1:
                self.source_lines = self.source_lines[:n_obs]
            self.pad_token_id = self.tokenizer.pad_token_id
            self.dataset_kwargs = dataset_kwargs
            dataset_kwargs.update({"add_prefix_space": True} if isinstance(self.tokenizer, BartTokenizer) else {})

    def standardlize(self, score):
        max_score = max(score)
        return [s-max_score for s in score]

    def __len__(self):
        return len(self.source_lines)

    @staticmethod
    def get_char_lens(sentences):
        return [len(x) for x in sentences]

    @cached_property
    def tgt_lens(self):
        """Length in characters of target documents"""
        return self.get_char_lens(self.tgt_file)

    def make_sortish_sampler(self, batch_size, distributed=False, shuffle=True, **kwargs):
        if distributed:
            return DistributedSortishSampler(self, batch_size, shuffle=shuffle, **kwargs)
        else:
            return SortishSampler(self.source_lines, batch_size, shuffle=shuffle)

    def make_dynamic_sampler(self, max_tokens_per_batch=1024, **kwargs):
        assert FAIRSEQ_AVAILABLE, "Dynamic batch size requires `pip install fairseq`"
        assert not self.used_char_len, "You must call  python make_len_file.py before calling make_dynamic_sampler"
        sorted_indices = list(self.make_sortish_sampler(1024, shuffle=False))

        def num_tokens_in_example(i):
            return min(self.source_lines[i], self.max_target_length)

        # call fairseq cython function
        batch_sampler: List[List[int]] = batch_by_size(
            sorted_indices,
            num_tokens_fn=num_tokens_in_example,
            max_tokens=max_tokens_per_batch,
            required_batch_size_multiple=64,
        )
        shuffled_batches = [batch_sampler[i] for i in np.random.permutation(range(len(batch_sampler)))]
        # move the largest batch to the front to OOM quickly (uses an approximation for padding)
        approximate_toks_per_batch = [max(self.source_lines[i] for i in batch) * len(batch) for batch in shuffled_batches]
        largest_batch_idx = np.argmax(approximate_toks_per_batch)
        shuffled_batches[0], shuffled_batches[largest_batch_idx] = (
            shuffled_batches[largest_batch_idx],
            shuffled_batches[0],
        )
        return shuffled_batches

    def __getitem__(self, item):
        raise NotImplementedError("You must implement this")

    def collate_fn(self, batch):
        raise NotImplementedError("You must implement this")

class Seq2SeqDataset(AbstractSeq2SeqDataset):
    """A dataset that calls prepare_seq2seq_batch."""

    def __getitem__(self, index) -> Dict[str, str]:
        index = index
        if self.dataset_kwargs['brio_loss']:
            source_line_H = self.prefix + self.source_lines[index]
            source_line_L = self.prefix + self.source_lines_L[index]
            tgt_line_H = self.tgt_lines_H[index]
            tgt_line_L = self.tgt_lines_L[index]
            manual_score_H = self.manualZ_H[index]
            manual_score_L = self.manualZ_L[index]

            assert source_line_H, f"empty source_H line for index {index}"
            assert source_line_L, f"empty source_L line for index {index}"
            assert tgt_line_H, f"empty tgt_H line for index {index}"
            assert tgt_line_L, f"empty tgt_L line for index {index}"

            return {"tgt_texts_H": tgt_line_H, "tgt_texts_L": tgt_line_L, \
                "src_texts_H": source_line_H, "src_texts_L": source_line_L, \
                "manual_score_H": manual_score_H, "manual_score_L": manual_score_L, "id": index}
        else:
            source_line = self.prefix + self.source_lines[index]
            tgt_line = self.tgt_lines[index]
            manual_score = self.manualZ[index]

            assert source_line, f"empty source line for index {index}"
            assert tgt_line, f"empty tgt line for index {index}"
            assert manual_score is not None, f"empty manual score for index {index}"

            return {"tgt_texts": tgt_line, "src_texts": source_line, "id": index, "manual_score": manual_score}


    def collate_fn(self, batch) -> Dict[str, torch.Tensor]:
        """Call prepare_seq2seq_batch."""
        batch_encoding: Dict[str, torch.Tensor] = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.max_source_length,
            max_target_length=self.max_target_length,
            return_tensors="pt",
            **self.dataset_kwargs,
        ).data
        batch_encoding["ids"] = torch.tensor([x["id"] for x in batch])
        return batch_encoding


class Seq2SeqDataCollator:
    def __init__(self, tokenizer, data_args, padding=None, tpu_num_cores=None, use_estimator=False):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.use_estimator = use_estimator
        assert (
            self.pad_token_id is not None
        ), f"pad_token_id is not defined for ({self.tokenizer.__class__.__name__}), it must be defined."
        self.data_args = data_args
        self.tpu_num_cores = tpu_num_cores
        self.dataset_kwargs = {"add_prefix_space": True} if isinstance(tokenizer, BartTokenizer) else {}
        if data_args.src_lang is not None:
            self.dataset_kwargs["src_lang"] = data_args.src_lang
        if data_args.tgt_lang is not None:
            self.dataset_kwargs["tgt_lang"] = data_args.tgt_lang
        self.padding = padding if padding is not None else ("max_length" if self.tpu_num_cores is not None else "longest")

    def __call__(self, batch) -> Dict[str, torch.Tensor]:
        assert hasattr(self.tokenizer, "prepare_seq2seq_batch"), f"tokenizer does not have prepare_seq2seq_batch"
        if self.data_args.brio_loss:
            manual_score_H = torch.tensor([x['manual_score_H'] for x in batch])
            manual_score_L = torch.tensor([x['manual_score_L'] for x in batch])

            batch_L = [{
                "tgt_texts": sample['tgt_texts_L'], 
                "src_texts": sample['src_texts_L'], 
                "id": sample['id'], 
            } for sample in batch]

            batch_H = [{
                "tgt_texts": sample['tgt_texts_H'], 
                "src_texts": sample['src_texts_H'], 
                "id": sample['id'], 
            } for sample in batch]

            batch_L = self._encode(batch_L)
            batch_H = self._encode(batch_H)

            input_ids_L, attention_mask_L, labels_L = (
                batch_L["input_ids"],
                batch_L["attention_mask"],
                batch_L["labels"],
            )    
            input_ids_H, attention_mask_H, labels_H = (
                batch_H["input_ids"],
                batch_H["attention_mask"],
                batch_H["labels"],
            )

            if isinstance(self.tokenizer, T5Tokenizer):
                decoder_input_ids_H = self._shift_right_t5(labels_H)
                decoder_input_ids_L = self._shift_right_t5(labels_L)
            else:
                decoder_input_ids_H = shift_tokens_right(labels_H, self.pad_token_id, self.decoder_start_token_id)
                decoder_input_ids_L = shift_tokens_right(labels_L, self.pad_token_id, self.decoder_start_token_id)

            batch = {
                "input_ids_L": input_ids_L,
                "attention_mask_L": attention_mask_L,
                "decoder_input_ids_L": decoder_input_ids_L,
                "labels_L": labels_L,
                "manual_score_L": manual_score_L,
                "input_ids_H": input_ids_H,
                "attention_mask_H": attention_mask_H,
                "decoder_input_ids_H": decoder_input_ids_H,
                "labels_H": labels_H,
                "manual_score_H": manual_score_H,
            }

        else:
            manual_score = torch.tensor([x["manual_score"] for x in batch])

            if self.data_args.direction == 'combine_direction'or self.use_estimator:
                batch_reverse = [{
                    "tgt_texts": sample['src_texts'], 
                    "src_texts": sample['tgt_texts'], 
                    "id": sample['id'], 
                } for sample in batch]

            batch = self._encode(batch)
            input_ids, attention_mask, labels = (
                batch["input_ids"],
                batch["attention_mask"],
                batch["labels"],
            )            
            if isinstance(self.tokenizer, T5Tokenizer):
                decoder_input_ids = self._shift_right_t5(labels)
            else:
                decoder_input_ids = shift_tokens_right(labels, self.pad_token_id, self.decoder_start_token_id)


            if self.data_args.direction == 'combine_direction' or self.use_estimator:
                batch_reverse = self._encode(batch_reverse)
                input_ids_reverse, attention_mask_reverse, labels_reverse = (
                    batch_reverse["input_ids"],
                    batch_reverse["attention_mask"],
                    batch_reverse["labels"],
                )            
                if isinstance(self.tokenizer, T5Tokenizer):
                    decoder_input_ids_reverse = self._shift_right_t5(labels_reverse)
                else:
                    decoder_input_ids_reverse = shift_tokens_right(labels_reverse, self.pad_token_id, self.decoder_start_token_id)


            if self.data_args.direction != 'combine_direction':
                if self.use_estimator:
                    batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "decoder_input_ids": decoder_input_ids,
                        "labels": labels,
                        "manual_score": manual_score,
                        "attention_mask_reverse": attention_mask_reverse,
                    }
                else:
                    batch = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "decoder_input_ids": decoder_input_ids,
                        "labels": labels,
                        "manual_score": manual_score,
                    }
            else:
                batch = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "decoder_input_ids": decoder_input_ids,
                    "labels": labels,
                    "input_ids_reverse": input_ids_reverse,
                    "attention_mask_reverse": attention_mask_reverse,
                    "decoder_input_ids_reverse": decoder_input_ids_reverse,
                    "labels_reverse": labels_reverse,
                    "manual_score": manual_score,
                }
            
        return batch

    def _shift_right_t5(self, input_ids):
        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = self.pad_token_id
        return shifted_input_ids

    def _encode(self, batch) -> Dict[str, torch.Tensor]:
        batch_encoding = self.tokenizer.prepare_seq2seq_batch(
            [x["src_texts"] for x in batch],
            tgt_texts=[x["tgt_texts"] for x in batch],
            max_length=self.data_args.max_source_length,
            max_target_length=self.data_args.max_target_length,
            padding=self.padding,  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        return batch_encoding.data
    
    def _bert_encode(self, batch):
        inputs = self.tokenizer(
            [x["src_texts"] for x in batch],
            truncation=True,
            max_length=self.data_args.max_source_length,
            padding=self.padding,  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )
        outputs = self.tokenizer(
            [x["tgt_texts"] for x in batch],
            truncation=True,
            max_length=self.data_args.max_target_length,
            padding=self.padding,  # TPU hack
            return_tensors="pt",
            **self.dataset_kwargs,
        )

        labels = outputs.input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        output_batch = {
            "input_ids" : inputs.input_ids,
            "attention_mask" : inputs.attention_mask,
            "decoder_input_ids": outputs.input_ids,
            "decoder_attention_mask": outputs.attention_mask,
            "labels": labels
        }

        return output_batch

class SortishSampler(Sampler):
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."

    def __init__(self, data, batch_size, shuffle=True):
        self.data, self.bs, self.shuffle = data, batch_size, shuffle

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sortish_sampler_indices(self.data, self.bs, shuffle=self.shuffle))


def sortish_sampler_indices(data: List, bs: int, shuffle=True) -> np.array:
    "Go through the text data by order of src length with a bit of randomness. From fastai repo."
    if not shuffle:
        return np.argsort(np.array(data) * -1)

    def key_fn(i):
        return data[i]

    idxs = np.random.permutation(len(data))
    sz = bs * 50
    ck_idx = [idxs[i : i + sz] for i in range(0, len(idxs), sz)]
    sort_idx = np.concatenate([sorted(s, key=key_fn, reverse=True) for s in ck_idx])
    sz = bs
    ck_idx = [sort_idx[i : i + sz] for i in range(0, len(sort_idx), sz)]
    max_ck = np.argmax([key_fn(ck[0]) for ck in ck_idx])  # find the chunk with the largest key,
    ck_idx[0], ck_idx[max_ck] = ck_idx[max_ck], ck_idx[0]  # then make sure it goes first.
    sort_idx = np.concatenate(np.random.permutation(ck_idx[1:])) if len(ck_idx) > 1 else np.array([], dtype=np.int)
    sort_idx = np.concatenate((ck_idx[0], sort_idx))
    return sort_idx


class DistributedSortishSampler(Sampler):
    """Copied from torch DistributedSampler"""

    def __init__(self, dataset, batch_size, num_replicas=None, rank=None, add_extra_examples=True, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        if add_extra_examples:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(dataset)
            self.num_samples = len(self.available_indices)
        self.batch_size = batch_size
        self.add_extra_examples = add_extra_examples
        self.shuffle = shuffle

    def __iter__(self) -> Iterable:
        g = torch.Generator()
        g.manual_seed(self.epoch)

        sortish_data = [self.dataset.source_lines[i] for i in self.available_indices]
        sortish_indices = sortish_sampler_indices(sortish_data, self.batch_size, shuffle=self.shuffle)
        indices = [self.available_indices[i] for i in sortish_indices]
        assert len(indices) == self.num_samples
        return iter(indices)

    @cached_property
    def available_indices(self) -> np.array:
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        available_indices = indices[self.rank : self.total_size : self.num_replicas]
        return available_indices

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


logger = getLogger(__name__)


def use_task_specific_params(model, task):
    """Update config with summarization specific params."""
    task_specific_params = model.config.task_specific_params

    if task_specific_params is not None:
        pars = task_specific_params.get(task, {})
        logger.info(f"setting model.config to task specific params for {task}:\n {pars}")
        logger.info("note: command line args may override some of these")
        model.config.update(pars)


def pickle_load(path):
    """pickle.load(path)"""
    with open(path, "rb") as f:
        return pickle.load(f)


def pickle_save(obj, path):
    """pickle.dump(obj, path)"""
    with open(path, "wb") as f:
        return pickle.dump(obj, f)


def flatten_list(summary_ids: List[List]):
    return [x for x in itertools.chain.from_iterable(summary_ids)]


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, sort_keys=True, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


ROUGE_KEYS = ["rouge1", "rouge2", "rougeL", "rougeLsum"]


def extract_rouge_mid_statistics(dct):
    new_dict = {}
    for k1, v1 in dct.items():
        mid = v1.mid
        new_dict[k1] = {stat: round(getattr(mid, stat), 4) for stat in ["precision", "recall", "fmeasure"]}
    return new_dict




# Utilities for freezing parameters and checking whether they are frozen


def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False


def freeze_embeds(model):
    """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
    model_type = model.config.model_type

    if model_type == "t5" or model_type == "mt5":
        freeze_params(model.shared)
        for d in [model.encoder, model.decoder]:
            freeze_params(d.embed_tokens)
    elif model_type == "fsmt":
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)
    else:
        freeze_params(model.model.shared)
        for d in [model.model.encoder, model.model.decoder]:
            freeze_params(d.embed_positions)
            freeze_params(d.embed_tokens)


def grad_status(model: nn.Module) -> Iterable:
    return (par.requires_grad for par in model.parameters())


def any_requires_grad(model: nn.Module) -> bool:
    return any(grad_status(model))


def assert_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    n_require_grad = sum(lmap(int, model_grads))
    npars = len(model_grads)
    assert not any(model_grads), f"{n_require_grad/npars:.1%} of {npars} weights require grad"


def assert_not_all_frozen(model):
    model_grads: List[bool] = list(grad_status(model))
    npars = len(model_grads)
    assert any(model_grads), f"none of {npars} weights require grad"


def parse_numeric_n_bool_cl_kwargs(unparsed_args: List[str]) -> Dict[str, Union[int, float, bool]]:
    """
    Parse an argv list of unspecified command line args to a dict.
    Assumes all values are either numeric or boolean in the form of true/false.
    """
    result = {}
    assert len(unparsed_args) % 2 == 0, f"got odd number of unparsed args: {unparsed_args}"
    num_pairs = len(unparsed_args) // 2
    for pair_num in range(num_pairs):
        i = 2 * pair_num
        assert unparsed_args[i].startswith("--")
        if unparsed_args[i + 1].lower() == "true":
            value = True
        elif unparsed_args[i + 1].lower() == "false":
            value = False
        else:
            try:
                value = int(unparsed_args[i + 1])
            except ValueError:
                value = float(unparsed_args[i + 1])  # this can raise another informative ValueError

        result[unparsed_args[i][2:]] = value
    return result


def write_txt_file(ordered_tgt, path):
    f = Path(path).open("w")
    for ln in ordered_tgt:
        f.write(ln + "\n")
        f.flush()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def check_output_dir(args, expected_items=0):
    """
    Checks whether to bail out if output_dir already exists and has more than expected_items in it

    `args`: needs to have the following attributes of `args`:
      - output_dir
      - do_train
      - overwrite_output_dir

    `expected_items`: normally 0 (default) - i.e. empty dir, but in some cases a few files are expected (e.g. recovery from OOM)
    """
    if (
        os.path.exists(args.output_dir)
        and len(os.listdir(args.output_dir)) > expected_items
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and "
            f"has {len(os.listdir(args.output_dir))} items in it (expected {expected_items} items). "
            "Use --overwrite_output_dir to overcome."
        )
