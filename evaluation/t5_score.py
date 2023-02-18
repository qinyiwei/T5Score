# %%
import torch
import torch.nn as nn
import traceback
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List
import numpy as np
from tqdm import tqdm
BART_LANG_DICT = {'en': 'en_XX',
                  'de': 'de_DE',
                  'es': 'es_XX',
                  'fr': 'fr_XX',
                  'id': 'id_ID',
                  'ru': 'ru_RU',
                  'tr': 'tr_TR',
                  'zh': 'zh_CN',
                  'pl': 'pl_PL',
                  'lt': 'lt_LT',
                  'gu': 'gu_IN',
                  'ja': 'ja_XX',
                  'kk': 'kk_KZ',
                  'cs': 'cs_CZ',
                  'fi': 'fi_FI',
                  'ta': 'ta_IN',
                  'iu': 'iu',
                  'ro': 'ro_RO',
                  'et': 'et_EE',
                  'ne': 'ne_NP',
                  'si': 'si_LK',
                  'iu': '<s>',
                  }


class T5Scorer:
    def __init__(self, device='cuda:0', max_length=1024,
                 checkpoint='google/mt5-base', src_lang='en',
                 tgt_lang='en', model_parallel=False):
        # Set up model
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False,
                                                       src_lang=BART_LANG_DICT[src_lang.lower()], tgt_lang=BART_LANG_DICT[tgt_lang.lower()])
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            checkpoint, from_tf=".ckpt" in checkpoint)
        self.model.eval()
        if model_parallel:
            self.model.parallelize()
        else:
            self.model.to(device)
        # Set up loss
        self.loss_fct = nn.NLLLoss(
            reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)

    def load(self, path):
        """ Load model from paraphrase finetuning """
        self.model.load_state_dict(torch.load(
            path, map_location=self.device))

    def score(self, srcs, tgts, batch_size):
        """ Score a batch of examples """
        score_list = []
        for i in tqdm(range(0, len(srcs), batch_size)):
            src_list = srcs[i: i + batch_size]
            tgt_list = tgts[i: i + batch_size]
            try:
                with torch.no_grad():
                    encoded_src = self.tokenizer(
                        src_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    encoded_tgt = self.tokenizer(
                        tgt_list,
                        max_length=self.max_length,
                        truncation=True,
                        padding=True,
                        return_tensors='pt'
                    )
                    src_tokens = encoded_src['input_ids'].to(self.device)
                    src_mask = encoded_src['attention_mask'].to(self.device)

                    tgt_tokens = encoded_tgt['input_ids'].to(self.device)
                    tgt_mask = encoded_tgt['attention_mask']
                    tgt_len = tgt_mask.sum(dim=1).to(self.device)

                    output = self.model(
                        input_ids=src_tokens,
                        attention_mask=src_mask,
                        labels=tgt_tokens
                    )
                    logits = output.logits.view(-1,
                                                self.model.config.vocab_size)
                    loss = self.loss_fct(self.lsm(logits), tgt_tokens.view(-1))
                    loss = loss.view(tgt_tokens.shape[0], -1)
                    loss = loss.sum(dim=1) / tgt_len
                    curr_score_list = [-x.item() for x in loss]
                    score_list += curr_score_list

            except RuntimeError:
                traceback.print_exc()
                print(f'source: {src_list}')
                print(f'target: {tgt_list}')
                exit(0)
        return score_list
