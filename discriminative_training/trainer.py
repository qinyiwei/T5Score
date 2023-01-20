from ast import Lambda
from logging import raiseExceptions
from transformers import  Seq2SeqTrainer
import torch.nn as nn
import torch.nn.functional as F

class SupMultiEvalTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        target = inputs.pop('manual_score')

        if self.args.direction == 'combine_direction':
            inputs_reverse = {}
            inputs_reverse['input_ids'] = inputs.pop('input_ids_reverse')
            inputs_reverse['attention_mask'] = inputs.pop('attention_mask_reverse')
            inputs_reverse['decoder_input_ids'] = inputs.pop('decoder_input_ids_reverse')
            inputs_reverse['labels'] = inputs.pop('labels_reverse')

        outputs = model(**inputs)

        if self.args.direction == 'combine_direction':
            outputs_reverse = model(**inputs_reverse)

        score_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        lsm = nn.LogSoftmax(dim=1)
        loss_fct = nn.MSELoss()

        tgt_len = inputs['attention_mask'].sum(dim=1)
        tgt_tokens = inputs['labels']
        logits = outputs.logits.view(-1, self.model.config.vocab_size)
        score = score_fct(lsm(logits), tgt_tokens.view(-1))
        score = score.view(tgt_tokens.shape[0], -1)
        score = -score.sum(dim=1) / tgt_len
        
        if self.args.direction == 'combine_direction':
            tgt_len_reverse = inputs_reverse['attention_mask'].sum(dim=1)
            tgt_tokens_reverse = inputs_reverse['labels']
            logits_reverse = outputs_reverse.logits.view(-1, self.model.config.vocab_size)
            score_reverse = score_fct(lsm(logits_reverse), tgt_tokens_reverse.view(-1))
            score_reverse = score_reverse.view(tgt_tokens_reverse.shape[0], -1)
            score_reverse = -score_reverse.sum(dim=1) / tgt_len_reverse
            score = (score + score_reverse)/2

        loss = loss_fct(score, target)
        return loss

class BrioTrainer(Seq2SeqTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_fct = nn.NLLLoss(reduction='none', ignore_index=self.model.config.pad_token_id)
        self.lsm = nn.LogSoftmax(dim=1)
        #self.loss = nn.MarginRankingLoss(margin=self.args.margin)

    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        inputs_H = {}
        inputs_H['input_ids'] = inputs.pop('input_ids_H')
        inputs_H['attention_mask'] = inputs.pop('attention_mask_H')
        inputs_H['decoder_input_ids'] = inputs.pop('decoder_input_ids_H')
        inputs_H['labels'] = inputs.pop('labels_H')

        inputs_L = {}
        inputs_L['input_ids'] = inputs.pop('input_ids_L')
        inputs_L['attention_mask'] = inputs.pop('attention_mask_L')
        inputs_L['decoder_input_ids'] = inputs.pop('decoder_input_ids_L')
        inputs_L['labels'] = inputs.pop('labels_L')

        outputs_H = model(**inputs_H)
        outputs_L = model(**inputs_L)


        tgt_len_H = inputs_H['attention_mask'].sum(dim=1)
        tgt_tokens_H = inputs_H['labels']
        logits_H = outputs_H.logits.view(-1, self.model.config.vocab_size)
        score_H = self.score_fct(self.lsm(logits_H), tgt_tokens_H.view(-1))
        score_H = score_H.view(tgt_tokens_H.shape[0], -1)
        score_H = -score_H.sum(dim=1) / tgt_len_H
        
        tgt_len_L = inputs_L['attention_mask'].sum(dim=1)
        tgt_tokens_L = inputs_L['labels']
        logits_L = outputs_L.logits.view(-1, self.model.config.vocab_size)
        score_L = self.score_fct(self.lsm(logits_L), tgt_tokens_L.view(-1))
        score_L = score_L.view(tgt_tokens_L.shape[0], -1)
        score_L = -score_L.sum(dim=1) / tgt_len_L

        loss = F.relu(score_L - score_H + self.args.margin*(inputs['manual_score_H']-inputs['manual_score_L']))
        loss = loss.mean()
        if self.args.loss_type=='hybrid_B':
            loss = self.args.loss_alpha * (outputs_H["loss"] + outputs_L["loss"]) + self.args.loss_beta * loss
        elif self.args.loss_type=='hybrid_H':
            loss = self.args.loss_alpha * outputs_H["loss"] + self.args.loss_beta * loss
        elif self.args.loss_type=='hybrid_L':
            loss = self.args.loss_alpha * outputs_L["loss"] + self.args.loss_beta * loss
        elif self.args.loss_type=='hybrid_H_filter':
            nll_loss = -score_H
            nll_loss = nll_loss[inputs['manual_score_H']>self.args.loss_th]
            loss = self.args.loss_alpha * (nll_loss.mean()) + self.args.loss_beta * loss
        else:
            if self.args.loss_type!='margin':
                raise NotImplementedError('unsupported loss_type:{}'.format(self.args.loss_type))
        #loss = self.loss(score_H, score_L, 1)

        return loss

class EstmMultiEvalTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        inputs['output_hidden_states'] = True
        target = inputs.pop('manual_score')

        if self.args.direction == 'combine_direction':
            inputs_reverse = {}
            inputs_reverse['input_ids'] = inputs.pop('input_ids_reverse')
            inputs_reverse['attention_mask'] = inputs.pop('attention_mask_reverse')
            inputs_reverse['decoder_input_ids'] = inputs.pop('decoder_input_ids_reverse')
            inputs_reverse['labels'] = inputs.pop('labels_reverse')

        score = model(**inputs)

        if self.args.direction == 'combine_direction':
            score_reverse = model(**inputs_reverse)
            score = (score + score_reverse)/2

        loss_fct = nn.MSELoss()


        loss = loss_fct(score, target)
        return loss