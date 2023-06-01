#!/bin/bash

set -e

BASE_DIR=/srv/scratch3/hauser/gender-neutral/data/evaluation/Frequency_Distribution
TERMINOLOGY=/srv/scratch3/hauser/gender-neutral/bachelorette/data/word_lists/genderapp_dump_2013-01-22_CC_BY-NC-SA.csv

MODELS=/srv/scratch3/hauser/gender-neutral/models
BASELINE1=$MODELS/baseline1
BASELINE2=$MODELS/baseline2
FILTERING1=$MODELS/filtering1
SUBSAMPLED=$MODELS/subsampled

GENDERY_COUNTS=/srv/scratch3/hauser/gender-neutral/data/evaluation/Frequency_Distribution/counts/gendery.counts.csv
TEMPLATES=/srv/scratch3/hauser/gender-neutral/data/evaluation/Frequency_Distribution/templates
BUCKETS_DIR=/srv/scratch3/hauser/gender-neutral/data/evaluation/Frequency_Distribution/buckets/baseline2
BUCKETS_B1=/srv/scratch3/hauser/gender-neutral/data/evaluation/Frequency_Distribution/buckets/baseline1
TESTSETS_DIR=/srv/scratch3/hauser/gender-neutral/data/evaluation/Frequency_Distribution/testsets

GENDERS=( m f )
NUMBERS=( SG PL )

TOOLS=/srv/scratch3/hauser/gender-neutral/bachelorette/tools
SCRIPTS=/srv/scratch3/hauser/gender-neutral/bachelorette/scripts
VENV=/srv/scratch3/hauser/gender-neutral/bachelorette/filter_venv

source $VENV/bin/activate

# Define random seed function
get_seeded_random() { seed="$1";  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null; }


# echo "Creating testset from templates."
# for gender in "${GENDERS[@]}"; do

#     echo "Filling templates for $gender templates."
#     testset_outdir=$TESTSETS_DIR/$gender
#     mkdir -p $testset_outdir $testset_outdir/final

#     for number in "${NUMBERS[@]}"; do
        
#         out=$testset_outdir/$number
#         mkdir -p $out

#         python3 $TOOLS/prepare_frequency_evaluation.py --indir $BUCKETS_DIR/$gender/$number \
#         --outdir $out \
#         --templates $TEMPLATES/$gender.$number.src
#     done

#     for bucket in 0 1 11 101 1001 10001; do
#         cat $testset_outdir/SG/*.$bucket.* $testset_outdir/PL/*.$bucket.* > $testset_outdir/final/$bucket.src
#     done        
# done

echo "Creating testset from templates."
for gender in "${GENDERS[@]}"; do

    echo "Filling templates for $gender templates."
    testset_outdir=$TESTSETS_DIR/baseline1/$gender
    mkdir -p $testset_outdir $testset_outdir/final

    for number in "${NUMBERS[@]}"; do
        
        out=$testset_outdir/$number
        mkdir -p $out

        python3 $TOOLS/prepare_frequency_evaluation.py --indir $BUCKETS_B1/$gender/$number \
        --outdir $out \
        --templates $TEMPLATES/$gender.$number.src
    done

    for bucket in 0 1 11 101 1001 10001; do
        cat $testset_outdir/SG/*.$bucket.* $testset_outdir/PL/*.$bucket.* > $testset_outdir/final/$bucket.src
    done        
done

echo "Done with creating testset."
# 
echo "Creating rewritings for each model."
for model in $BASELINE1 ; do # $BASELINE2 $FILTERING1 $SUBSAMPLED

    model_name=$(basename "$model")
    echo "Creating Rewritings for $model_name"

    model_dir=$BASE_DIR/$model_name
    mkdir -p $model_dir

    for gender in "${GENDERS[@]}"; do

        rewritings_out=$model_dir/own_buckets/$gender
        mkdir -p $rewritings_out

        for bucket in 0 1 11 101 1001 10001; do

            bash $SCRIPTS/rewrite.sh $model/train_data $TESTSETS_DIR/baseline1/$gender/final $bucket.src $model/model $rewritings_out $bucket.trg $TOOLS 7
            
        done
    done
done

echo "Done."

