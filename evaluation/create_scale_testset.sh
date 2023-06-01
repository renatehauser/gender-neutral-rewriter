#!/bin/bash

set -e

GEN_DATA=/srv/scratch3/hauser/gender-neutral/data/de/raw/raw.gn.gen

MODELS=/srv/scratch3/hauser/gender-neutral/models
BASELINE1=$MODELS/baseline2
BASELINE2=$MODELS/baseline1
FILTERING1=$MODELS/filtering1
SUBSAMPLED=$MODELS/subsampled

WRANGLING=/srv/scratch3/hauser/gender-neutral/data/evaluation/Scale/wrangling
OUTDIR=/srv/scratch3/hauser/gender-neutral/data/evaluation/Scale/testset

TOOLS=/srv/scratch3/hauser/gender-neutral/bachelorette/tools
VENV=/srv/scratch3/hauser/gender-neutral/bachelorette/filter_venv

source $VENV/bin/activate

# Define random seed function
get_seeded_random() { seed="$1";  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null; }

echo "Subsample 50'000 from gendered segments."
shuf --random-source=<(get_seeded_random 1234) $GEN_DATA | head -n 50000 > $WRANGLING/50k.gen.src

echo "Run opusfilter on subsample of gendered segments."
opusfilter $WRANGLING/filter_gen_raw.yaml


cp $WRANGLING/filtered.gen.src $WRANGLING/tmp.gen.src
cp $WRANGLING/filtered.gen.src $WRANGLING/tmp.gen.trg

for model in $BASELINE1 $BASELINE2 $FILTERING1 $SUBSAMPLED; do
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

echo "Subsample 10'000 of the cleaned segments and copy to final testdata dir."
shuf --random-source=<(get_seeded_random 1234) $WRANGLING/cleaned.gen.src | head -n 10000 > $OUTDIR/test.scale.src

echo "Done."

