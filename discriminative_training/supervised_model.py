from transformers import T5ForConditionalGeneration
import torch
import torch.nn as nn
from typing import List, Optional, Dict

from torch.nn import Parameter, ParameterList

class EstimatorT5(T5ForConditionalGeneration):
    def __init__(self, config, dropout=0.1, layer_norm=False, hidden_sizes: List[int] = [768], \
            use_last_hidden_stats=False, use_token_pooling=False):
        super().__init__(config)
        self.config = config
        if not use_last_hidden_stats:
            self.layerwise_attention = LayerwiseAttention(
                    num_layers=config.num_layers + 1,
                    dropout=dropout,
                    layer_norm=layer_norm,
                )
        self.use_last_hidden_stats = use_last_hidden_stats
        self.use_token_pooling = use_token_pooling
        #self.estimator = FeedForward(
        #    in_dim=config.d_model * 6,
        #    hidden_sizes=hidden_sizes,
        #)

        self.estimator = FeedForward(
            in_dim=config.d_model,
            hidden_sizes=hidden_sizes,
        )

    def forward(
        self,
        **kwargs,
    ):
        attention_mask_reverse = kwargs.pop('attention_mask_reverse')
        outputs = super().forward(**kwargs)
        if self.use_token_pooling:
            #src_emb = self.compute_sentence_embedding(kwargs['input_ids'], outputs['encoder_hidden_states'], kwargs['attention_mask'])
            tgt_emb = self.compute_sentence_embedding(kwargs['labels'], outputs['decoder_hidden_states'], attention_mask_reverse, 
                        self.use_last_hidden_stats)
            #score = self.estimate(src_emb, tgt_emb, src_emb)
            score = self.estimator(tgt_emb)
        else:
            tgt_emb = self.compute_token_embedding(outputs['decoder_hidden_states'], attention_mask_reverse, 
                        self.use_last_hidden_stats)
            score = self.estimator(tgt_emb)
            score = score.sum(1)
        score = score.squeeze(-1)
        return score

    def estimate(
        self,
        src_sentemb: torch.Tensor,
        mt_sentemb: torch.Tensor,
        ref_sentemb: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb

        embedded_sequences = torch.cat(
            (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src),
            dim=1,
        )
        score = self.estimator(embedded_sequences)
        return score

    def compute_token_embedding(
        self, hidden_stats: torch.Tensor, attention_mask: torch.Tensor,
        use_last_hidden_stats: bool
    ) -> torch.Tensor:

        if use_last_hidden_stats:
            embeddings = hidden_stats[-1]
        else:
            if not self.training:
                n_splits = len(torch.split(hidden_stats[-1], 8))
                embeddings = []
                for split in range(n_splits):
                    all_layers = []
                    for layer in range(len(hidden_stats)):
                        layer_embs = torch.split(hidden_stats[layer], 8)
                        all_layers.append(layer_embs[split])
                    split_attn = torch.split(attention_mask, 8)[split]
                    embeddings.append(self.layerwise_attention(all_layers, split_attn))
                embeddings = torch.cat(embeddings, dim=0)
            else:
                embeddings = self.layerwise_attention(
                    hidden_stats, attention_mask
                )
        return embeddings

    def compute_sentence_embedding(
        self, input_ids: torch.Tensor, hidden_stats: torch.Tensor, attention_mask: torch.Tensor,
        use_last_hidden_stats: bool
    ) -> torch.Tensor:

        embeddings = self.compute_token_embedding(hidden_stats, attention_mask, use_last_hidden_stats)
        sentemb = average_pooling(
            input_ids,
            embeddings,
            attention_mask,
            self.config.pad_token_id,
        )
        return sentemb

def mask_fill(
    fill_value: float,
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """
    Function that masks embeddings representing padded elements.
    :param fill_value: the value to fill the embeddings belonging to padded tokens.
    :param tokens: The input sequences [bsz x seq_len].
    :param embeddings: word embeddings [bsz x seq_len x hiddens].
    :param padding_index: Index of the padding token.
    """
    padding_mask = tokens.eq(padding_index).unsqueeze(-1)
    return embeddings.float().masked_fill_(padding_mask, fill_value).type_as(embeddings)

def average_pooling(
    tokens: torch.Tensor,
    embeddings: torch.Tensor,
    mask: torch.Tensor,
    padding_index: int,
) -> torch.Tensor:
    """Average pooling function.
    :param tokens: Word ids [batch_size x seq_length]
    :param embeddings: Word embeddings [batch_size x seq_length x hidden_size]
    :param mask: Padding mask [batch_size x seq_length]
    :param padding_index: Padding value.
    """
    wordemb = mask_fill(0.0, tokens, embeddings, padding_index)
    sentemb = torch.sum(wordemb, 1)
    sum_mask = mask.unsqueeze(-1).expand(embeddings.size()).float().sum(1)
    return sentemb / sum_mask




class LayerwiseAttention(torch.nn.Module):
    def __init__(
        self,
        num_layers: int,
        layer_norm: bool = False,
        layer_weights: Optional[List[int]] = None,
        dropout: float = None,
    ) -> None:
        super(LayerwiseAttention, self).__init__()
        self.num_layers = num_layers
        self.layer_norm = layer_norm
        self.dropout = dropout

        if layer_weights is None:
            layer_weights = [0.0] * num_layers
        elif len(layer_weights) != num_layers:
            raise Exception(
                "Length of layer_weights {} differs \
                from num_layers {}".format(
                    layer_weights, num_layers
                )
            )

        self.scalar_parameters = ParameterList(
            [
                Parameter(
                    torch.FloatTensor([layer_weights[i]]),
                    requires_grad=True,
                )
                for i in range(num_layers)
            ]
        )

        self.alpha = Parameter(torch.FloatTensor([1.0]), requires_grad=True)

        if self.dropout:
            dropout_mask = torch.zeros(len(self.scalar_parameters))
            dropout_fill = torch.empty(len(self.scalar_parameters)).fill_(-1e20)
            self.register_buffer("dropout_mask", dropout_mask)
            self.register_buffer("dropout_fill", dropout_fill)

    def forward(
        self,
        tensors: List[torch.Tensor],  # pylint: disable=arguments-differ
        mask: torch.Tensor = None,
    ) -> torch.Tensor:

        if len(tensors) != self.num_layers:
            raise Exception(
                "{} tensors were passed, but the module was initialized to \
                mix {} tensors.".format(
                    len(tensors), self.num_layers
                )
            )

        def _layer_norm(tensor, broadcast_mask, num_elements_not_masked):
            tensor_masked = tensor * broadcast_mask
            mean = torch.sum(tensor_masked) / num_elements_not_masked
            variance = (
                torch.sum(((tensor_masked - mean) * broadcast_mask) ** 2)
                / num_elements_not_masked
            )
            return (tensor - mean) / torch.sqrt(variance + 1e-12)

        # BUG: Pytorch bug fix when Parameters are not well copied across GPUs
        # https://github.com/pytorch/pytorch/issues/36035
        if len([parameter for parameter in self.scalar_parameters]) != self.num_layers:
            weights = torch.tensor(self.weights, device=tensors[0].device)
            alpha = torch.tensor(self.alpha, device=tensors[0].device)
        else:
            weights = torch.cat([parameter for parameter in self.scalar_parameters])
            alpha = self.alpha

        if self.training and self.dropout:
            weights = torch.where(
                self.dropout_mask.uniform_() > self.dropout, weights, self.dropout_fill
            )

        normed_weights = torch.nn.functional.softmax(weights, dim=0)
        normed_weights = torch.split(normed_weights, split_size_or_sections=1)

        if not self.layer_norm:
            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(weight * tensor)
            return alpha * sum(pieces)

        else:
            mask_float = mask.float()
            broadcast_mask = mask_float.unsqueeze(-1)
            input_dim = tensors[0].size(-1)
            num_elements_not_masked = torch.sum(mask_float) * input_dim

            pieces = []
            for weight, tensor in zip(normed_weights, tensors):
                pieces.append(
                    weight
                    * _layer_norm(tensor, broadcast_mask, num_elements_not_masked)
                )
            return alpha * sum(pieces)


class FeedForward(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_sizes: List[int] = [3072, 768],
        activations: str = "Sigmoid",
        final_activation: Optional[str] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        modules = []
        modules.append(nn.Linear(in_dim, hidden_sizes[0]))
        modules.append(self.build_activation(activations))
        modules.append(nn.Dropout(dropout))

        for i in range(1, len(hidden_sizes)):
            modules.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            modules.append(self.build_activation(activations))
            modules.append(nn.Dropout(dropout))

        modules.append(nn.Linear(hidden_sizes[-1], int(out_dim)))
        if final_activation is not None:
            modules.append(self.build_activation(final_activation))

        self.ff = nn.Sequential(*modules)

    def build_activation(self, activation: str) -> nn.Module:
        if hasattr(nn, activation.title()):
            return getattr(nn, activation.title())()
        else:
            raise Exception(f"{activation} is not a valid activation function!")

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        return self.ff(in_features)