#!/usr/bin/env python3

'''
Splits Moses-style bitext into test and train sets.
OR
Removes segments belonging to the testset from the train data.
'''

import random
from argparse import ArgumentParser


def filtering(src, trg, src_out, trg_out, tst_src, tst_trg):
    c = 0
    for src_seg, trg_seg in zip(src, trg):
        if src_seg in tst_src or trg_seg in tst_trg:
            c += 1
        else:
            src_out.write(src_seg)
            trg_out.write(trg_seg)
    print (" ".join(["Removed", str(c), "Segments"]))


def split(src, trg, src_out, trg_out, src_test, trg_test):
    assert (len(src) == len(trg))
    segments = list(zip(src, trg))
    random.shuffle(segments)
    c = 0
    for segment in segments:
        if c < args.testsize:
            src_test.write(segment[0])
            trg_test.write(segment[1])
            c += 1
        else:
            src_out.write(segment[0])
            trg_out.write(segment[1])


def main(args):
    with open(args.input_files + "." + args.src) as src_in, open(args.input_files + "." + args.trg) as trg_in:
        src = src_in.readlines()
        trg = trg_in.readlines()

    print(len(src), len(trg))
    
    with open(args.testset + "." + args.src) as src_testset, open(args.testset + "." + args.trg) as trg_testset, open(args.output + "." + args.src, "w") as src_out, open(args.output + "." + args.trg, "w") as trg_out: 
        #testset = set(zip(open(args.testset + "." + args.src).readlines(), open(args.testset + "." + args.trg).readlines()))
        tst_src = set(src_testset.readlines())
        tst_trg = set(trg_testset.readlines())
        filtering(src, trg, src_out, trg_out, tst_src, tst_trg)


def get_parser():
    parser = ArgumentParser('Splits parallel files into train, dev and test set.')
    parser.add_argument('--input_files', type=str, help='Input files base name; language code will be used as suffix.')
    parser.add_argument('--testset', type=str, default='__no_testset__', help='If a testset is already available, segments from the testset will be stripped from the training data (input_files)')
    parser.add_argument('-t', '--testsize', type=int, default=3000, help='Number of segments in train set.')
    parser.add_argument('--output', type=str, default='corpus', help='The prefix for the output files')
    parser.add_argument('--src', type=str, default='en', help='Source language code.')
    parser.add_argument('--trg', type=str, default='de', help='Target language code.')
    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)