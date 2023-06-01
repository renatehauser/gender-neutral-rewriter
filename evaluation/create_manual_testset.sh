#!/bin/bash

set -e

BASE_DIR=/srv/scratch3/hauser/gender-neutral

GEN_DATA=$BASE_DIR/data/de/raw/raw.gn.gen

MODELS=$BASE_DIR/models
BASELINE1=$MODELS/baseline2
BASELINE2=$MODELS/baseline1
FILTERING1=$MODELS/filtering1
SUBSAMPLED=$MODELS/subsampled
GENDERY=$MODELS/gendery

EVAL_DIR=$BASE_DIR/data/evaluation/Manual
WRANGLING=$EVAL_DIR/wrangling
OUTDIR=$EVAL_DIR/testset

TOOLS=$BASE_DIR/bachelorette/tools
VENV=$BASE_DIR/bachelorette/filter_venv

source $VENV/bin/activate

# # Define random seed function
get_seeded_random() { seed="$1";  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null; }

echo "Subsample 10'000 randomly from gendered segments."
shuf --random-source=<(get_seeded_random 1234) $GEN_DATA | tail -n +100000000  | head -n 10000 > $WRANGLING/10k.gen.src


echo "Run opusfilter on subsample of gendered segments."
opusfilter $WRANGLING/filter_gen_raw.yaml


cp $WRANGLING/filtered.gen.src $WRANGLING/tmp.gen.src
cp $WRANGLING/filtered.gen.src $WRANGLING/tmp.gen.trg

for model in $BASELINE1 $BASELINE2 $FILTERING1 $SUBSAMPLED $GENDERY; do
    echo "Removing segments from $model training data"
    python3 $TOOLS/remove_testset.py --input_files $WRANGLING/tmp.gen \
    --testset $model/train_data/train \
    --output $WRANGLING/cleaned.gen \
    --src src \
    --trg trg
    # feed output of filtering into next filtering step
    mv $WRANGLING/cleaned.gen.src $WRANGLING/tmp.gen.src
    mv $WRANGLING/cleaned.gen.trg $WRANGLING/tmp.gen.trg
done

mv $WRANGLING/tmp.gen.src $WRANGLING/cleaned.gen.src
mv $WRANGLING/tmp.gen.trg $WRANGLING/cleaned.gen.trg

echo "Subsample 300 of the cleaned segments and copy to final testdata dir."
shuf --random-source=<(get_seeded_random 1234) $WRANGLING/cleaned.gen.src | head -n 300 > $OUTDIR/test.manual.src

echo "Done."

