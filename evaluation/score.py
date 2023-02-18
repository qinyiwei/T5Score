import argparse
import time
import numpy as np
from utils import *
import csv
import json


class Scorer:
    """ Support T5Score """

    def __init__(self, file_path, device='cuda:0', model_type=None,
                 src_lang='en', tgt_lang='en', score_type='segment', batch_size=None,
                 t5score_ref="ref", is_MQM=False, no_layer_norm=False, model_parallel=False):
        """ file_path: path to the pickle file
            All the data are normal capitalized, not tokenied, including src, ref, sys
        """
        self.device = device
        self.model_type = model_type
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.lp = self.src_lang+'-'+self.tgt_lang
        self.score_type = score_type
        self.batch_size = batch_size
        self.output_detailed = {}
        self.t5score_ref = t5score_ref
        self.is_MQM = is_MQM
        self.no_layer_norm = no_layer_norm
        self.model_parallel = model_parallel

        self.SYSName = []
        self.SEGID = []
        self.TestSet = []
        self.srcs = []
        self.refs = []
        self.syss = []

        if not is_MQM:
            self.manualRaw = []
            self.manualZ = []
        else:
            self.mqmScore = []
            self.error = []

        if file_path.endswith('.csv'):
            file_manual = open(file_path)
            csvreader = csv.reader(file_manual, delimiter=' ')
            header = next(csvreader)

            for row in csvreader:
                self.SYSName.append(row[header.index('SYSName')])
                self.SEGID.append(row[header.index('SEGID')])
                self.TestSet.append(row[header.index('TestSet')])
                self.srcs.append(row[header.index('src')])
                self.refs.append(row[header.index('ref')])
                if self.tgt_lang == 'zh':
                    self.syss.append(row[header.index('sys')].replace(" ", ""))
                else:
                    self.syss.append(row[header.index('sys')])
                if not is_MQM:
                    self.manualRaw.append(row[header.index('manualRaw')])
                    self.manualZ.append(row[header.index('manualZ')])
                else:
                    self.mqmScore.append(row[header.index('mqmScore')])
                    self.error.append(row[header.index('error')])

        elif file_path.endswith('.json'):
            data = json.load(open(file_path))
            header = data[0]
            for row in data[1:]:
                self.SYSName.append(row[header.index('SYSName')])
                self.SEGID.append(row[header.index('SEGID')])
                self.TestSet.append(row[header.index('TestSet')])
                self.srcs.append(row[header.index('src')])
                self.refs.append(row[header.index('ref')])
                if self.lp == 'en-zh':
                    self.syss.append(row[header.index('sys')].replace(" ", ""))
                else:
                    self.syss.append(row[header.index('sys')])
                if not is_MQM:
                    self.manualRaw.append(row[header.index('manualRaw')])
                    self.manualZ.append(row[header.index('manualZ')])
                else:
                    self.mqmScore.append(row[header.index('mqmScore')])
                    self.error.append(row[header.index('error')])
        else:
            raise NotImplementedError("File type not surpported!")

        print(f'Data loaded from {file_path}.')

    def save_data(self, path):
        if not self.is_MQM:
            header = ['SYSName', 'SEGID', 'TestSet', 'src',
                      'ref', 'sys', 'manualRaw', 'manualZ']
        else:
            header = ['SYSName', 'SEGID', 'TestSet',
                      'src', 'ref', 'sys', 'mqmScore', 'error']
        for key in self.all_scores.keys():
            header.append(key)

        if path.endswith('.csv'):
            with open(path, 'w') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerow(header)
                for i in range(len(self.SYSName)):
                    if not self.is_MQM:
                        row = [self.SYSName[i], self.SEGID[i], self.TestSet[i], self.srcs[i],
                               self.refs[i], self.syss[i], self.manualRaw[i], self.manualZ[i]]
                    else:
                        row = [self.SYSName[i], self.SEGID[i], self.TestSet[i], self.srcs[i],
                               self.refs[i], self.syss[i], self.mqmScore[i], self.error[i]]
                    for key in self.all_scores.keys():
                        row.append(self.all_scores[key][i])
                    writer.writerow(row)
        elif path.endswith('.json'):
            output_data = []
            output_data.append(header)
            for i in range(len(self.SYSName)):
                if not self.is_MQM:
                    row = [self.SYSName[i], self.SEGID[i], self.TestSet[i], self.srcs[i],
                           self.refs[i], self.syss[i], self.manualRaw[i], self.manualZ[i]]
                else:
                    row = [self.SYSName[i], self.SEGID[i], self.TestSet[i], self.srcs[i],
                           self.refs[i], self.syss[i], self.mqmScore[i], self.error[i]]
                for key in self.all_scores.keys():
                    row.append(self.all_scores[key][i])
                output_data.append(row)
            with open(path, 'w') as f:
                json.dump(output_data, f)
        else:
            raise NotImplementedError("File type not surpported!")

        print("save file to: "+path)

    def score(self, metrics):
        self.all_scores = {}
        for metric_name in metrics:
            if metric_name == 't5_score':
                from t5_score import T5Scorer

                def run_t5score(scorer, mt: list, ref: list):
                    hypo_ref = np.array(scorer.score(mt, ref,
                                                     batch_size=4 if self.batch_size is None else self.batch_size))
                    ref_hypo = np.array(scorer.score(ref, mt,
                                                     batch_size=4 if self.batch_size is None else self.batch_size))
                    return ref_hypo, hypo_ref

                # Set up T5Score
                t5_scorer = T5Scorer(device=self.device, checkpoint=self.model_type,
                                     src_lang=self.src_lang, tgt_lang=self.tgt_lang, model_parallel=self.model_parallel)

                start = time.time()
                print(f'Begin calculating T5Score.')
                if self.t5score_ref == "ref":
                    scores_precision, scores_recall = run_t5score(
                        t5_scorer, self.syss, self.refs)
                    self.all_scores[metric_name +
                                    "_ref_precision"] = scores_precision
                    self.all_scores[metric_name+"_ref_recall"] = scores_recall
                    self.all_scores[metric_name +
                                    "_ref_F"] = (scores_precision + scores_recall)/2
                elif self.t5score_ref == "src":
                    scores_precision, scores_recall = run_t5score(
                        t5_scorer, self.syss, self.srcs)
                    self.all_scores[metric_name +
                                    "_src_precision"] = scores_precision
                    self.all_scores[metric_name+"_src_recall"] = scores_recall
                    self.all_scores[metric_name +
                                    "_src_F"] = (scores_precision + scores_recall)/2
                elif self.t5score_ref == "ref_src":
                    scores_precision1, scores_recall1 = run_t5score(
                        t5_scorer, self.syss, self.refs)
                    scores_precision2, scores_recall2 = run_t5score(
                        t5_scorer, self.syss, self.srcs)
                    self.all_scores[metric_name +
                                    "_ref_precision"] = scores_precision1
                    self.all_scores[metric_name+"_ref_recall"] = scores_recall1
                    self.all_scores[metric_name +
                                    "_src_precision"] = scores_precision2
                    self.all_scores[metric_name+"_src_recall"] = scores_recall2
                    self.all_scores[metric_name+"_ref_src_F"] = (scores_precision1 +
                                                                 scores_recall1 + scores_precision2 + scores_recall2)/4
                else:
                    raise NotImplementedError(
                        "t5score_ref {} not implemented!".format(self.t5score_ref))

                print(
                    f'Finished calculating T5Score, time passed {time.time() - start}s.')

            else:
                raise NotImplementedError(
                    "Metric {} not implemented!".format(metric_name))


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--file', type=str, required=True,
                        help='The data to load from.')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='The device to run on.')
    parser.add_argument('--output', type=str, required=True,
                        help='The output path to save the calculated scores.')
    parser.add_argument('--t5_score', action='store_true', default=False,
                        help='Whether to calculate T5Score')
    parser.add_argument('--model_score', action='store_true', default=False,
                        help='Whether to calculate model score')
    parser.add_argument('--model_type', default='google/mt5-base', type=str, required=False,
                        help='The LM used by T5Score.')
    parser.add_argument('--src_lang', default='en', type=str, required=False,
                        help='The source language.')
    parser.add_argument('--tgt_lang', default='en', type=str, required=False,
                        help='The target language.')
    parser.add_argument('--score_type', default='segment', type=str, required=False,
                        help='Could be segment|system|document')
    parser.add_argument('--batch_size', default=None, type=int, required=False,
                        help='Batch size used by pretrained LM.')
    parser.add_argument('--save_detailed', action='store_true', default=False,
                        help='Whether to save detailed analysis info.')

    parser.add_argument('--t5score_ref', default='ref', type=str, required=False,
                        help='Could be ref|src|ref_src|src_ref')
    parser.add_argument('--is_MQM', action='store_true', default=False,
                        help='Whether to use MQM score.')
    parser.add_argument('--no_layer_norm', action='store_true', default=False,
                        help='Whether to use layer_norm.')
    parser.add_argument('--model_parallel', action='store_true', default=False,
                        help="If there are more than one devices, whether to use model parallelism to distribute the model's modules across devices.")

    args = parser.parse_args()

    scorer = Scorer(args.file, args.device, args.model_type,
                    args.src_lang, args.tgt_lang, args.score_type, args.batch_size,
                    args.t5score_ref, args.is_MQM, args.no_layer_norm, args.model_parallel)

    METRICS = []
    if args.t5_score:
        METRICS.append('t5_score')

    print("t5score_ref:{}".format(args.t5score_ref))
    print("outptut:{}".format(args.output))

    dir = os.path.dirname(args.output)
    if not os.path.exists(dir):
        os.makedirs(dir)

    scorer.score(METRICS)
    scorer.save_data(args.output)


if __name__ == '__main__':
    main()
