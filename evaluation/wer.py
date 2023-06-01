import csv
import sys
from typing import List

import numpy as np
from argparse import ArgumentParser, FileType

from jiwer import wer
from sacremoses import MosesTokenizer

np.random.seed(0)
mtok = MosesTokenizer(lang='en')


def get_parser() -> ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = ArgumentParser('Run automatic evaluation.')
    parser.add_argument('-r', '--reference',
                        required=True,
                        type=FileType('r'),
                        help='File with reference sentences.')
    parser.add_argument('--hypotheses',
                        nargs="+",
                        type=FileType('r'),
                        help='File with first model outputs to consider.')
    parser.add_argument('-m', '--models',
                        nargs='+',
                        help='Model names corresponding to the hypothesis files.')
    parser.add_argument('-o', '--csv-outfile',
                        type=str,
                        help='CSV file to write the stats to.')
    parser.add_argument('--header',
                        nargs='*',
                        help='Header row for the output csv.')
    parser.add_argument('-n', '--num_samples',
                        default=1000,
                        type=int,
                        help='The number of bootstrap samples to draw.')
    parser.add_argument('-s', '--sample_ratio',
                        default=1,
                        type=float,
                        help='The ratio of samples to draw in each iteration.')
    parser.add_argument('--test-significance', action='store_true', help='Test significance of WER of two hypotheses. If set, only provide 2 hypotheses!')
    return parser


def paired_bootstrap_resampling(ref: List[str],
                                hyp1: List[str],
                                hyp2: List[str],
                                num_samples: int = 1000,
                                sample_ratio: float = 1) -> None:
    '''
    Compute statistical significance with paired boostrap sampling and
    print results.

    Based on https://github.com/neubig/util-scripts/blob/master/paired-bootstrap.py
    '''

    assert(len(ref) == len(hyp1))
    assert(len(ref) == len(hyp1))

    hyp1_scores = []
    hyp2_scores = []
    wins = [0, 0, 0]
    n = len(ref)
    ids = list(range(n))

    for _ in range(num_samples):

        # Subsample the testsets
        reduced_ids = np.random.choice(ids,int(len(ids)*sample_ratio),replace=True)
        reduced_ref = [ref[i] for i in reduced_ids]
        reduced_hyp1 = [hyp1[i] for i in reduced_ids]
        reduced_hyp2 = [hyp2[i] for i in reduced_ids]

        # Calculate WER on the sample and save stats
        hyp1_score = wer(reduced_ref, reduced_hyp1)
        hyp2_score = wer(reduced_ref, reduced_hyp2)

        if hyp1_score < hyp2_score:
            wins[0] += 1
        elif hyp1_score > hyp2_score:
            wins[1] += 1
        else:
            wins[2] += 1
        hyp1_scores.append(hyp1_score)
        hyp2_scores.append(hyp2_score)

    # Print win stats
    wins = [x/float(num_samples) for x in wins]
    print('Win ratio: hyp1=%.3f, hyp2=%.3f, tie=%.3f' % (wins[0], wins[1], wins[2]))
    if wins[0] > wins[1]:
        print('(hyp1 is superior with p value p=%.3f)\n' % (1-wins[0]), flush=True)
    elif wins[1] > wins[0]:
        print('(hyp2 is superior with p value p=%.3f)\n' % (1-wins[1]), flush=True)

    # Print system stats
    hyp1_scores.sort()
    hyp2_scores.sort()
    print('hyp1 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
         (np.mean(hyp1_scores), np.median(hyp1_scores), hyp1_scores[int(num_samples * 0.025)], hyp1_scores[int(num_samples * 0.975)]), flush=True)
    print('hyp2 mean=%.3f, median=%.3f, 95%% confidence interval=[%.3f, %.3f]' %
         (np.mean(hyp2_scores), np.median(hyp2_scores), hyp2_scores[int(num_samples * 0.025)], hyp2_scores[int(num_samples * 0.975)]), flush=True)


if __name__ == '__main__':
    '''
    Compute WER and statistical significance.
    '''

    parser = get_parser()
    args = parser.parse_args()

    ref_lines = [' '.join(mtok.tokenize(i)) for i in args.reference]

    if args.test_significance:
        hyp1_lines = [' '.join(mtok.tokenize(i)) for i in args.hypotheses[0]]
        hyp2_lines = [' '.join(mtok.tokenize(i)) for i in args.hypotheses[1]]

        paired_bootstrap_resampling(ref_lines, hyp1_lines, hyp2_lines, args.num_samples, args.sample_ratio)


    else:

        with open(args.csv_outfile, 'w', encoding='utf-8') as csv_out:
            writer = csv.writer(csv_out, delimiter=',')
            if args.header:
                writer.writerow(args.header)


            for model, hyp in zip(args.models, args.hypotheses):
                hyp_lines = [' '.join(mtok.tokenize(i)) for i in hyp]
                result = round(wer(ref_lines, hyp_lines) * 100, 2)
                writer.writerow([model, result])
