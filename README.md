# T5Score
This is the Repo for the paper: [T5Score: Discriminative Fine-tuning of Generative Evaluation Metrics](https://arxiv.org/abs/2212.05726).

## Evaluation
### Calculate Evaluation Score
You can run T5Score on a corpus to get automatic evaluation score. An [example](evaluation/examples/generative/run_wmt20_mt5_large_paracotta.sh) to evaluate on WMT20 is provided. We also provide an example [result](evaluation/results/scores.csv) of language pair cs-en for corpus WMT20.

### Calculate Correlation Score
To compare the automatic evaluation method with human judgements, you can run segment level analysis and system level analysis as follows:
```bash
$ python evaluation/calculate_corr.py \
      --dir "./evaluation/results/" \
      --language_pair "cs-en" \
      --filename "scores.csv" \
      --metrics "t5_score_ref_F"
```


## Training
### Generative Training
You can use parallel data to train your custom unsupervised T5Score. An [example](generative_training/examples/train_mt5_large_paracotta.sh) trained on a multilingual paraphrase dataset Paracotta is provided.

### Discriminative Training
You can use paired data with human judgements to train your custom supervised T5Score. An [example](discriminative_training/examples/train_mt5_large_wmt17_19.sh) trained on dataset from WMT17 to WMT19 is provided.
