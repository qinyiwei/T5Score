from analysis import StatSeg, StatSys
import os
import argparse


def print_correlation_result(dir, pair, filename, metrics):
    print("***************************************segment level***************************************")
    file = os.path.join(os.path.join(dir, pair), filename)
    if os.path.exists(file):
        print('Language pair is: '+pair)
        wmt_stat = StatSeg(path=file, metrics=metrics, is_MQM=False)
        wmt_stat.print_segment_corr(corr="kendalltau_like")

    print("***************************************system level***************************************")
    file = os.path.join(os.path.join(dir, pair), filename)
    if os.path.exists(file):
        print('Language pair is: '+pair)
        wmt_stat = StatSys(path=file, metrics=metrics, is_MQM=False)
        wmt_stat.print_system_corr(corr="kendalltau")
        wmt_stat.print_system_corr(corr="pearson")


def main():
    parser = argparse.ArgumentParser(description='Scorer parameters')
    parser.add_argument('--dir', type=str, required=True,
                        help='The directory of evaluation file.')
    parser.add_argument('--language_pair', type=str, required=True,
                        help='Language pair.')
    parser.add_argument('--filename', type=str, required=True,
                        help='The name of evaluation file.')
    parser.add_argument('--metrics', type=str, nargs='+', required=False,
                        default=['t5_score_ref_F'],
                        help='The automatic evaluation metrics used to calculate correlation with human scores.')
    args = parser.parse_args()

    print_correlation_result(
        args.dir, args.language_pair, args.filename, args.metrics)


if __name__ == '__main__':
    main()
