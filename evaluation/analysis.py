# %%
from utils import *
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
import csv
from datasets import load_dataset


class StatSeg:
    """ A class used to get stats of WMT trained data """

    def __init__(self, path, metrics=None,
                 is_MQM=False, MQM_path=None, threshold=25, no_Human=True, pair=None):
        self.path = path
        self.metrics = metrics
        if self.metrics is None:
            with open(path, 'r') as f:
                csvreader = csv.reader(f, delimiter=' ')
                header = next(csvreader)
                self.metrics = header[8:]

        self.is_MQM = is_MQM
        self.threshold = threshold
        self.no_Human = no_Human
        self.pair = pair

        if is_MQM:
            self.MQM = self.read_MQM(MQM_path)

    def read_MQM(self, MQM_path):
        file = open(MQM_path)
        csvreader = csv.reader(file, delimiter=' ')
        header = next(csvreader)
        MQM = {}
        for row in csvreader:
            SYSName = row[header.index('SYSName')]
            SEGID = row[header.index('SEGID')]
            mqmScore = float(row[header.index('mqmScore')])
            if SYSName not in MQM:
                MQM[SYSName] = {}
            MQM[SYSName][SEGID] = mqmScore
        return MQM

    def read_csv(self, path, propertyName):
        file = open(path)
        csvreader = csv.reader(file, delimiter=' ')
        header = next(csvreader)
        scores = {}
        num = 0
        for row in csvreader:
            SYSName = row[header.index('SYSName')]
            SEGID = row[header.index('SEGID')]
            score = row[header.index(propertyName)]
            score = float(score) if score != '' else None

            if not self.is_MQM:
                scoreManual = row[header.index('manualRaw')]
                scoreManual = float(scoreManual) if scoreManual != '' else None
            else:
                if SYSName not in self.MQM:
                    continue
                scoreManual = self.MQM[SYSName][SEGID] if SEGID in self.MQM[SYSName] else None

            if self.no_Human and ('Human' in SYSName or 'HUMAN' in SYSName or SYSName.startswith('ref')):
                continue

            if scoreManual is None:
                continue

            if SEGID not in scores:
                scores[SEGID] = []
            scores[SEGID].append([scoreManual, score])
            num += 1

        print("property:{}, number:{}".format(propertyName, num))
        return scores

    def count(self, score):
        conc = 0
        disc = 0
        num = 0
        for i in range(1, len(score)):
            for j in range(0, i):
                if abs(score[i][0]-score[j][0]) < self.threshold or abs(score[i][0]-score[j][0]) == 0:
                    continue
                # system i is better than system j
                elif score[i][0]-score[j][0] >= self.threshold:
                    if score[i][1] > score[j][1]:
                        conc += 1
                    else:
                        disc += 1
                    num += 1
                else:  # system i is worse than system j
                    if score[i][1] < score[j][1]:
                        conc += 1
                    else:
                        disc += 1
                    num += 1
        return conc, disc, num

    def kendalltau_like(self, scores):
        totalSegNum = 0
        totalConc = 0
        totalDisc = 0

        for score in scores.values():
            conc, disc, num = self.count(score)
            totalSegNum += num
            totalConc += conc
            totalDisc += disc

        print("totalConc:{}, totalDisc:{}, totalSegNum:{}".format(
            totalConc, totalDisc, totalSegNum))
        return (totalConc - totalDisc) / (totalConc + totalDisc)

    def print_segment_corr(self, metrics=None, corr="kendalltau_like"):
        headers = ['metric', 'k-tau']
        metric_at_segment_level = []
        self.segScore = {}
        if metrics is None:
            metrics = self.metrics

        for metric in metrics:
            scores = self.read_csv(self.path, metric)
            if corr == "kendalltau_like":
                segScore = self.kendalltau_like(scores)
            else:
                raise NotImplementedError
            metric_at_segment_level.append([metric, segScore])
            self.segScore[metric] = segScore
        sorted_metric_at_segment_level = sorted(
            metric_at_segment_level, key=lambda x: x[1], reverse=True)
        print(tabulate(sorted_metric_at_segment_level,
                       headers=headers, tablefmt='simple'))


class StatSys:
    """ A class used to get stats of WMT trained data """

    def __init__(self, path, metrics=None, manual='manualZ', is_MQM=False, MQM_path=None,
                 no_Human=True, outlier=None, pair=None):
        self.path = path
        self.metrics = metrics
        self.no_Human = no_Human
        self.pair = pair

        if self.metrics is None:
            with open(path, 'r') as f:
                csvreader = csv.reader(f, delimiter=' ')
                header = next(csvreader)
                self.metrics = header[8:]

        if is_MQM:
            self.manual_data = self.read_MQM(MQM_path)
        else:
            self.manual_data = self.read_csv(self.path, manual)

        if outlier is not None:
            import pandas as pd
            self.outlier = pd.read_pickle(outlier)
        else:
            self.outlier = None

    def read_MQM(self, MQM_path):
        file = open(MQM_path)
        csvreader = csv.reader(file, delimiter=' ')
        header = next(csvreader)
        MQM = {}
        for row in csvreader:
            SYSName = row[header.index('SYSName')]
            SEGID = row[header.index('SEGID')]
            mqmScore = float(row[header.index('mqmScore')])
            if SYSName not in MQM:
                MQM[SYSName] = []
            MQM[SYSName].append(mqmScore)
        return MQM

    def read_csv(self, path, propertyName):
        file = open(path)
        csvreader = csv.reader(file, delimiter=' ')
        header = next(csvreader)
        scores = {}
        num = 0
        for row in csvreader:
            systemName = row[header.index('SYSName')]
            score = row[header.index(propertyName)]
            score = float(score) if score != '' else None

            if systemName not in scores:
                scores[systemName] = []
            if score is not None:
                scores[systemName].append(score)
                num += 1

        print("property:{},number:{}".format(propertyName, num))

        return scores

    def retrieve_scores(self, data, keys):
        """ retrieve better, worse scores """
        systemScore = []
        for sys in keys:
            if self.outlier is not None and sys in self.outlier[self.pair]:
                continue
            systemScore.append(sum(data[sys])/len(data[sys]))
        return systemScore

    def pearson(self, hyp1_scores: list, hyp2_scores: list):
        """ Computes the system level pearson correlation score. """
        assert len(hyp1_scores) == len(hyp2_scores)
        return pearsonr(hyp1_scores, hyp2_scores)[0], len(hyp1_scores)

    def kendalltau(self, hyp1_scores: list, hyp2_scores: list):
        """ Computes the system level kendalltau correlation score. """
        assert len(hyp1_scores) == len(hyp2_scores)
        return kendalltau(hyp1_scores, hyp2_scores)[0], len(hyp1_scores)

    def spearmanr(self, hyp1_scores: list, hyp2_scores: list):
        """ Computes the system level spearmanr correlation score. """
        assert len(hyp1_scores) == len(hyp2_scores)
        return spearmanr(hyp1_scores, hyp2_scores)[0], len(hyp1_scores)

    def print_system_corr(self, metrics=None, corr="pearson"):
        headers = ['metric', 'pearson']
        metric_at_system_level = []
        self.systemScore = {}
        if metrics is None:
            metrics = self.metrics

        for metric in tqdm(metrics):
            self.auto_data = self.read_csv(self.path, metric)
            keys = [i for i in self.manual_data.keys(
            ) if i in self.auto_data.keys()]
            if self.no_Human:
                keys = [key for key in keys if (
                    'Human' not in key and 'HUMAN' not in key and (not key.startswith('ref')))]
            if not self.no_Human:
                keys = [key for key in keys if 'Human-A.0' not in key]

            manual = self.retrieve_scores(self.manual_data, keys)
            system = self.retrieve_scores(self.auto_data, keys)
            if corr == "pearson":
                systemScore = self.pearson(system, manual)
            elif corr == "kendalltau":
                systemScore = self.kendalltau(system, manual)
            elif corr == "spearmanr":
                systemScore = self.spearmanr(system, manual)
            else:
                raise NotImplementedError

            metric_at_system_level.append([metric, systemScore])
            self.systemScore[metric] = systemScore[0]
        sorted_metric_at_system_level = sorted(
            metric_at_system_level, key=lambda x: x[1], reverse=True)
        print(tabulate(sorted_metric_at_system_level,
                       headers=headers, tablefmt='simple'))


class WMTStatQE:
    """ A class used to get stats of WMT trained data """

    def __init__(self, path, submission_file=None, metrics=None, is_submission=False, threshold=25):
        self.path = path
        self.submission_file = submission_file
        self.metrics = metrics
        if not is_submission:
            if self.metrics is None:
                with open(path, 'r') as f:
                    csvreader = csv.reader(f, delimiter=' ')
                    header = next(csvreader)
                    self.metrics = header[8:]

        self.threshold = threshold
        self.is_submission = is_submission
        self.language_pair = path.split("/")[1]

    def count(self, score):
        conc = 0
        disc = 0
        num = 0
        for i in range(1, len(score)):
            for j in range(0, i):
                if abs(score[i][0]-score[j][0]) < self.threshold or abs(score[i][0]-score[j][0]) == 0:
                    continue
                # system i is better than system j
                elif score[i][0]-score[j][0] >= self.threshold:
                    if score[i][1] > score[j][1]:
                        conc += 1
                    else:
                        disc += 1
                    num += 1
                else:  # system i is worse than system j
                    if score[i][1] < score[j][1]:
                        conc += 1
                    else:
                        disc += 1
                    num += 1
        return conc, disc, num

    def kendall(self, scores):
        totalSegNum = 0
        totalConc = 0
        totalDisc = 0

        for score in scores.values():
            conc, disc, num = self.count(score)
            totalSegNum += num
            totalConc += conc
            totalDisc += disc

        print("totalConc:{}, totalDisc:{}, totalSegNum:{}".format(
            totalConc, totalDisc, totalSegNum))
        return (totalConc - totalDisc) / (totalConc + totalDisc)

    def print_ktau(self, metrics=None):
        headers = ['metric', 'k-tau']
        metric_with_ktau = []
        self.ktauScore = {}
        if metrics is None:
            metrics = self.metrics

        for metric in tqdm(metrics):
            scores = self.read_csv(self.path, metric)
            ktauScore = self.kendall(scores)
            metric_with_ktau.append([metric, ktauScore])
            self.ktauScore[metric] = ktauScore
        sorted_metric_with_ktau = sorted(
            metric_with_ktau, key=lambda x: x[1], reverse=True)
        print(tabulate(sorted_metric_with_ktau,
                       headers=headers, tablefmt='simple'))

    def print_pearsonr(self, metrics=None):
        headers = ['metric', 'k-tau']
        metric_with_pearsonr = []
        self.pearsonrScore = {}
        if metrics is None:
            metrics = self.metrics

        if not self.is_submission:
            scores_manual = self.read_csv_qe(self.path, 'manualZ')
        else:
            if self.language_pair == 'ru-en':
                data = load_dataset(self.path.split(
                    '/')[0], self.path.split('/')[1], split="validation")
            else:
                data = load_dataset(self.path.split(
                    '/')[0], self.path.split('/')[1], split="test")

            print(f'Data loaded from {self.path}.')
            SEGID = []
            scores_manual = []
            for sample in data:
                SEGID.append(sample['segid'])
                scores_manual.append(sample['z_mean'])
                # scores_manual.append(sample['mean'])

        for metric in tqdm(metrics):
            if not self.is_submission:
                scores = self.read_csv_qe(self.path, metric)
            else:
                scores = self.read_txt_qe(self.submission_file)
            #scores = stats.zscore(scores)
            pearsonrScore = pearsonr(scores_manual, scores)[0]
            metric_with_pearsonr.append([metric, pearsonrScore])
            self.pearsonrScore[metric] = pearsonrScore
        sorted_metric_with_ktau = sorted(
            metric_with_pearsonr, key=lambda x: x[1], reverse=True)
        print(tabulate(sorted_metric_with_ktau,
                       headers=headers, tablefmt='simple'))

    def read_csv_qe(self, path, propertyName):
        file = open(path)
        csvreader = csv.reader(file, delimiter=' ')
        header = next(csvreader)
        scores = []
        for row in csvreader:
            score = row[header.index(propertyName)]
            score = float(score) if score != '' else None
            if score is not None:
                scores.append(score)

        return scores

    def read_txt_qe(self, path):
        file = open(path)
        scores = []
        for row in file:
            line = row[:-1].split('\t')
            score = float(line[3]) if line[3] != '' else None
            scores.append(score)
        return scores
