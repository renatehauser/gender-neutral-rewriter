#!/bin/bash

BASE_DIR=/srv/scratch3/hauser/gender-neutral/data/evaluation/Copy
#TERMINOLOGY=/srv/scratch3/hauser/gender-neutral/bachelorette/data/word_lists/genderapp_dump_2013-01-22_CC_BY-NC-SA.csv
OUT_PATH=$BASE_DIR/stats
TESTSET=/srv/scratch3/hauser/gender-neutral/data/evaluation/Copy/testset/test.copy.src

VENV=/srv/scratch3/hauser/gender-neutral/bachelorette/filter_venv

source $VENV/bin/activate

B1=$BASE_DIR/baseline1/baseline1.copy.trg
B2=$BASE_DIR/baseline2/baseline2.copy.trg
F1=$BASE_DIR/filtering1/filtering1.copy.trg
S=$BASE_DIR/subsampled/subsampled.copy.trg
G=$BASE_DIR/gendery/gendery.copy.trg

python3 -m tools.wer --reference $TESTSET \
--hypotheses $B1 $B2 $F1 $S $G \
--models baseline1 baseline2 filtering1 subsampled gendery \
--csv-outfile $OUT_PATH/wer.csv \
--header model WER