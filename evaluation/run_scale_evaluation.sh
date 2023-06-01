#!/bin/bash

BASE_DIR=/srv/scratch3/hauser/gender-neutral/data/evaluation/Scale
TERMINOLOGY=/srv/scratch3/hauser/gender-neutral/bachelorette/data/word_lists/genderapp_dump_2013-01-22_CC_BY-NC-SA.csv
OUT_PATH=$BASE_DIR/stats

VENV=/srv/scratch3/hauser/gender-neutral/bachelorette/filter_venv

source $VENV/bin/activate

for  model in baseline1 baseline2 filtering1 subsampled; do

    python3 -m tools.exact_match_accuracy --source-path $BASE_DIR/testset \
    --target-path $BASE_DIR/$model \
    --terminology-path $TERMINOLOGY \
    --model $model \
    --gn \
    --out-file $OUT_PATH/$model.csv \
    --header $model accuracy

done

python3 -m tools.exact_match_accuracy --source-path $BASE_DIR/testset \
--target-path $BASE_DIR/gendery \
--terminology-path $TERMINOLOGY \
--model gendery \
--gf \
--out-file $OUT_PATH/gendery.csv \
--header $model accuracy

python3 -m tools.aggregate_results --results-path $OUT_PATH \
--output-path $OUT_PATH/stats.all.csv \
--header modelname accuracy