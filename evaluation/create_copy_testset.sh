#!/bin/bash

set -e

UNM_DATA=/srv/scratch3/hauser/gender-neutral/data/de/raw/raw.filtered.gn.unmatched
NEUT_DATA=/srv/scratch3/hauser/gender-neutral/data/de/raw/filtered_splits/reordered/not_parallel/raw.80M.filtered.gn.neut.00.11 # these segments are for sure in no training data


MODELS=/srv/scratch3/hauser/gender-neutral/models
BASELINE1=$MODELS/baseline2
BASELINE2=$MODELS/baseline1
FILTERING1=$MODELS/filtering1
SUBSAMPLED=$MODELS/subsampled

WRANGLING=/srv/scratch3/hauser/gender-neutral/data/evaluation/Copy/wrangling
OUTDIR=/srv/scratch3/hauser/gender-neutral/data/evaluation/Copy/testset

TOOLS=/srv/scratch3/hauser/gender-neutral/bachelorette/tools
VENV=/srv/scratch3/hauser/gender-neutral/bachelorette/filter_venv

source $VENV/bin/activate

# Define random seed function
get_seeded_random() { seed="$1";  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null; }


# subsample 2000 unmatched segments from somewhere in the middle of the file to avoid occurrences in the training data
total_lines=$(wc -l < $UNM_DATA)
start_line=$((total_lines / 4))
num_lines=$((total_lines / 2))

echo "Subsample 2000 unm and neut segments"
tail -n +"$((start_line + 1))" $UNM_DATA | head -n "$num_lines" | shuf --random-source=<(get_seeded_random 1234) | head -n 2000 > $WRANGLING/2k.unm.src
# subsample 2000 neut segments
shuf --random-source=<(get_seeded_random 1234) $NEUT_DATA | head -n 2000 > $WRANGLING/2k.neut.src

cp $WRANGLING/2k.neut.src $WRANGLING/tmp.2k.neut.src
cp $WRANGLING/2k.neut.src $WRANGLING/tmp.2k.neut.trg

cp $WRANGLING/2k.neut.src $WRANGLING/tmp.2k.unm.src
cp $WRANGLING/2k.neut.src $WRANGLING/tmp.2k.unm.trg

# TODO: run remove_testset against training data of all models
for model in $BASELINE1 $BASELINE2 $FILTERING1 $SUBSAMPLED; do
    echo "Removing segments from $model training data"
    python3 $TOOLS/remove_testset.py --input_files $WRANGLING/tmp.2k.neut \
    --testset $model/train_data/train \
    --output $WRANGLING/cleaned.2k.neut \
    --src src \
    --trg trg
    # feed output of filtering into next filtering step
    mv $WRANGLING/cleaned.2k.neut.src $WRANGLING/tmp.2k.neut.src
    mv $WRANGLING/cleaned.2k.neut.trg $WRANGLING/tmp.2k.neut.trg

    python3 $TOOLS/remove_testset.py --input_files $WRANGLING/tmp.2k.unm \
    --testset $model/train_data/train \
    --output $WRANGLING/cleaned.2k.unm \
    --src src \
    --trg trg
    # feed output of filtering into next filtering step
    mv $WRANGLING/cleaned.2k.unm.src $WRANGLING/tmp.2k.unm.src
    mv $WRANGLING/cleaned.2k.unm.trg $WRANGLING/tmp.2k.unm.trg
done

mv $WRANGLING/tmp.2k.neut.src $WRANGLING/cleaned.2k.neut.src
mv $WRANGLING/tmp.2k.neut.trg $WRANGLING/cleaned.2k.neut.trg

mv $WRANGLING/tmp.2k.unm.src $WRANGLING/cleaned.2k.unm.src
mv $WRANGLING/tmp.2k.unm.trg $WRANGLING/cleaned.2k.unm.trg


echo "from the filtered, subsample 1000 each"
shuf --random-source=<(get_seeded_random 1234) $WRANGLING/cleaned.2k.neut.src | head -n 1000 > $WRANGLING/final.neut.src
shuf --random-source=<(get_seeded_random 1234) $WRANGLING/cleaned.2k.unm.src | head -n 1000 > $WRANGLING/final.unm.src

echo "cat and copy final testset"
cat $WRANGLING/final.neut.src $WRANGLING/final.unm.src > $OUTDIR/test.copy.src
cp $OUTDIR/test.copy.src $OUTDIR/test.copy.trg

echo "Done."