#!/bin/bash

set -e

# Set variables
DATA_DIR=/scratch/reni/gn/filtering1/train_data
OUTPUT_DIR=/scratch/reni/gn/filtering1/model_filtering1
VENV_DIR=/scratch/reni/gn/baseline2/venv
REPO_DIR=/scratch/reni/gn/bachelorette
GPUS=$5

SCRIPTS_DIR=$REPO_DIR/scripts
TOOLS_DIR=$REPO_DIR/tools


python3 -m venv --prompt=filtering1 venv 
source venv/bin/activate
pip install sentencepiece

git clone https://github.com/awslabs/sockeye.git
cd sockeye && pip3 install --editable . && cd ..

# Define random seed function
get_seeded_random() { seed="$1";  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt </dev/zero 2>/dev/null; }


# subsample training data for the sentencepiece model
shuf -n 2000000 --random-source=<(get_seeded_random 1234) $DATA_DIR/train.src > $DATA_DIR/subsampled.train.src
shuf -n 2000000 --random-source=<(get_seeded_random 1234) $DATA_DIR/train.trg > $DATA_DIR/subsampled.train.trg

# Preprocess data
echo "Training SentencePiece model..."
python3 $TOOLS_DIR/spm_train.py --input $DATA_DIR/subsampled.train.src $DATA_DIR/subsampled.train.trg \
          --model-prefix $DATA_DIR/spm \
          --vocab-size 8000 \
          --character-coverage 1 \
          --model-type bpe \
          --shuffle-input-sentence True \
          --user-defined-symbols @@GFM@@

python3 $SCRIPTS_DIR/convert_spm_vocab.py $DATA_DIR/spm.vocab

echo "Preprocessing trainset..."
python3 $TOOLS_DIR/spm_encode.py --model $DATA_DIR/spm.model --infile $DATA_DIR/train.src --outfile $DATA_DIR/train.bpe.src
python3 $TOOLS_DIR/spm_encode.py --model $DATA_DIR/spm.model --infile $DATA_DIR/train.trg --outfile $DATA_DIR/train.bpe.trg

echo "Shuffling processed trainsets..."
shuf --random-source=<(get_seeded_random 1234) $DATA_DIR/train.bpe.src > $DATA_DIR/shuf.train.bpe.src
shuf --random-source=<(get_seeded_random 1234) $DATA_DIR/train.bpe.trg > $DATA_DIR/shuf.train.bpe.trg

echo "Preprocessing validset..."
python3 $TOOLS_DIR/spm_encode.py --model $DATA_DIR/spm.model --infile $DATA_DIR/valid.src --outfile $DATA_DIR/valid.bpe.src
python3 $TOOLS_DIR/spm_encode.py --model $DATA_DIR/spm.model --infile $DATA_DIR/valid.trg --outfile $DATA_DIR/valid.bpe.trg


# Prepare training data
python3 -m sockeye.prepare_data \
--source-vocab $DATA_DIR/spm.vocab.json \
--target-vocab $DATA_DIR/spm.vocab.json \
-s $DATA_DIR/shuf.train.bpe.src \
-t $DATA_DIR/shuf.train.bpe.trg  \
--pad-vocab-to-multiple-of 8 --bucket-width 8 --shared-vocab \
--max-seq-len 143 \
--max-processes 20 \
-o $DATA_DIR/sockeye_train_data_prepared


# Run training
PARAMS=$OUTPUT_DIR/params.best

echo "Training..."

# CUDA_VISIBLE_DEVICES=$GPUS 
torchrun --no_python --nproc_per_node 4 sockeye-train \
    --prepared-data $DATA_DIR/sockeye_train_data_prepared \
    --validation-source $DATA_DIR/valid.bpe.src \
    --validation-target $DATA_DIR/valid.bpe.trg \
    --output $OUTPUT_DIR \
    --num-embed 512 \
    --num-layers=6:6 \
    --transformer-model-size 512 \
    --transformer-attention-heads 4 \
    --transformer-feed-forward-num-hidden 1024 \
    --amp \
    --batch-type max-word \
    --batch-size 10000 \
    --update-interval 1 \
    --checkpoint-interval 500 \
    --optimizer-betas 0.9:0.98 \
    --dist \
    --initial-learning-rate 0.004 \
    --learning-rate-scheduler-type inv-sqrt-decay \
    --learning-rate-warmup 4000 \
    --max-num-checkpoint-not-improved=8 \
    --seed 1 \
    --decode-and-evaluate -1 \
    --max-num-epochs 30 \
    --keep-last-params=2 \
    --max-seq-len=143 \
    --quiet-secondary-workers \
    --env=PYTORCH_JIT=0


echo "Training done..."

# Exit success
exit 0