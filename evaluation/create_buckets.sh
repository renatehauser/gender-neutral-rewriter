#!/bin/bash


BACHELORETTE=/srv/scratch3/hauser/gender-neutral/bachelorette
TOOLS_DIR=$BACHELORETTE/tools
TERMINOLOGY=$BACHELORETTE/data/word_lists/genderapp_dump_2013-01-22_CC_BY-NC-SA.csv
COUNTS_DIR=/srv/scratch3/hauser/gender-neutral/data/evaluation/Frequency_Distribution/counts
BUCKETS_DIR=/srv/scratch3/hauser/gender-neutral/data/evaluation/Frequency_Distribution/buckets
# define paths
GENDERY_DATA=/srv/scratch3/hauser/gender-neutral/data/evaluation/Gendery/combined.gf.src
BASELINE_DATA=/srv/scratch3/hauser/gender-neutral/models/baseline2/train_data/train.src
BASELINE1_DATA=/srv/scratch3/hauser/gender-neutral/models/baseline1/train_data/train.src
FILTERING1_DATA=/srv/scratch3/hauser/gender-neutral/models/filtering1/train_data/train.src
SUBSAMPLED_DATA=/srv/scratch3/hauser/gender-neutral/models/subsampled/train_data/train.src

source /srv/scratch3/hauser/gender-neutral/bachelorette/filter_venv/bin/activate


# run counting on Chanti's Data
python3 $TOOLS_DIR/frequencies.py --inpath $GENDERY_DATA \
 --terminology $TERMINOLOGY \
 --match-level lemma \
 --count $COUNTS_DIR/gendery.counts.csv \
 --count-only


echo "Run counting on baseline2 data"
python3 $TOOLS_DIR/frequencies.py --inpath $BASELINE_DATA \
 --terminology $TERMINOLOGY \
 --match-level lemma \
 --count $COUNTS_DIR/baseline2.counts.csv \
 --count-only

echo "Run counting on baseline1 data"
python3 $TOOLS_DIR/frequencies.py --inpath $BASELINE1_DATA \
 --terminology $TERMINOLOGY \
 --match-level lemma \
 --count $COUNTS_DIR/baseline1.counts.csv \
 --count-only

echo "Run counting on filtering1 data"
python3 $TOOLS_DIR/frequencies.py --inpath $FILTERING1_DATA \
 --terminology $TERMINOLOGY \
 --match-level lemma \
 --count $COUNTS_DIR/filtering1.counts.csv \
 --count-only

echo "Run counting on subsampled data"
python3 $TOOLS_DIR/frequencies.py --inpath $FILTERING1_DATA \
 --terminology $TERMINOLOGY \
 --match-level lemma \
 --count $COUNTS_DIR/subsampled.counts.csv \
 --count-only


# create buckets for M and F (check counts list: does it make sense to have SG and PL separately??)
GENDERS=( m f )
NUMBERS=( SG PL )

echo "Create buckets for baseline2."
for gender in "${GENDERS[@]}"; do
    for number in "${NUMBERS[@]}"; do
        python3 $TOOLS_DIR/create_buckets.py --counts $COUNTS_DIR/gendery.counts.csv $COUNTS_DIR/baseline2.counts.csv \
        --gender $gender \
        --number $number \
        --outprefix $BUCKETS_DIR/baseline2/bucket.gendery.baseline2
    done
done

echo "Create buckets for baseline1."
for gender in "${GENDERS[@]}"; do
    for number in "${NUMBERS[@]}"; do
        python3 $TOOLS_DIR/create_buckets.py --counts $COUNTS_DIR/gendery.counts.csv $COUNTS_DIR/baseline1.counts.csv \
        --gender $gender \
        --number $number \
        --outprefix $BUCKETS_DIR/baseline1/bucket.gendery.baseline1
    done
done

echo "Create buckets for filtering1."
for gender in "${GENDERS[@]}"; do
    for number in "${NUMBERS[@]}"; do
        python3 $TOOLS_DIR/create_buckets.py --counts $COUNTS_DIR/gendery.counts.csv $COUNTS_DIR/filtering1.counts.csv \
        --gender $gender \
        --number $number \
        --outprefix $BUCKETS_DIR/filtering1/bucket.gendery.filtering1
    done
done

echo "Create buckets for subsampled."
for gender in "${GENDERS[@]}"; do
    for number in "${NUMBERS[@]}"; do
        python3 $TOOLS_DIR/create_buckets.py --counts $COUNTS_DIR/gendery.counts.csv $COUNTS_DIR/subsampled.counts.csv \
        --gender $gender \
        --number $number \
        --outprefix $BUCKETS_DIR/subsampled/bucket.gendery.subsampled
    done
done



