#!/bin/bash

set -e

BASE_DIR=/srv/scratch3/hauser/gender-neutral/data/evaluation/Frequency_Distribution
TERMINOLOGY=/srv/scratch3/hauser/gender-neutral/bachelorette/data/word_lists/genderapp_dump_2013-01-22_CC_BY-NC-SA.csv
OUT_PATH=$BASE_DIR/stats
PLOTS_PATH=$BASE_DIR/plots

# MODELS=/srv/scratch3/hauser/gender-neutral/models
# BASELINE1=$MODELS/baseline1
# BASELINE2=$MODELS/baseline2
# FILTERING1=$MODELS/filtering1
# SUBSAMPLED=$MODELS/subsampled

GENDERS=( m f )

TOOLS=/srv/scratch3/hauser/gender-neutral/bachelorette/tools
SCRIPTS=/srv/scratch3/hauser/gender-neutral/bachelorette/scripts
VENV=/srv/scratch3/hauser/gender-neutral/bachelorette/filter_venv

source $VENV/bin/activate


echo "Evaluating Backward-Replacement"
for gender in "${GENDERS[@]}"; do

    python3 -m tools.exact_match_accuracy --source-path $BASE_DIR/testsets/baseline1/$gender/final \
    --target-path $BASE_DIR/Backward-Replacement/own_buckets/$gender \
    --terminology-path $TERMINOLOGY \
    --gender $gender \
    --model Backward-Replacement \
    --gn \
    --out-file $OUT_PATH/$gender/Backward-Replacement.$gender.csv

done

MODELS=( Round-Trip-Translation Term-Based-Filtering Subsampled )

for model_name in "${MODELS[@]}"; do # $BASELINE1 $BASELINE2 $FILTERING1 $SUBSAMPLED

    #model_name=$(basename "$model")
    echo "Evaluating $model_name"

    for gender in "${GENDERS[@]}"; do

        mkdir -p $OUT_PATH/$gender

        python3 -m tools.exact_match_accuracy --source-path $BASE_DIR/testsets/$gender/final \
        --target-path $BASE_DIR/$model_name/$gender \
        --terminology-path $TERMINOLOGY \
        --gender $gender \
        --model $model_name \
        --gn \
        --out-file $OUT_PATH/$gender/$model_name.$gender.csv

    done

done

echo "Evaluating gendery"
for gender in "${GENDERS[@]}"; do

    python3 -m tools.exact_match_accuracy --source-path $BASE_DIR/testsets/$gender/final \
    --target-path $BASE_DIR/Bias-to-Debias/$gender \
    --sort-files \
    --terminology-path $TERMINOLOGY \
    --gender $gender \
    --model gendery \
    --gf \
    --out-file $OUT_PATH/$gender/Bias-to-Debias.$gender.csv

done


for gender in "${GENDERS[@]}"; do

    echo "Aggregate results and plot."
    python3 -m tools.aggregate_results --results-path $OUT_PATH/$gender \
    --output-path $OUT_PATH/stats.all.$gender.csv \
    --plot $PLOTS_PATH/plot.all.$gender.png \
    --frequency-analysis \
    --header modelname gender bucket_0 bucket_1 bucket_11 bucket_101 bucket_1001 bucket_10001

done
