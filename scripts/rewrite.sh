#!/bin/bash

DATA_DIR=$1
IN_DIR=$2
IN_NAME=$3
MODEL_DIR=$4
OUT_DIR=$5
OUT_NAME=$6
TOOLS_DIR=$7
GPU=$8


IN_FILE=$IN_DIR/$IN_NAME
IN_ENCODED=$IN_DIR/bpe
OUT_FILE=$OUT_DIR/$OUT_NAME
OUT_ENCODED=$OUT_DIR/bpe
mkdir -p $IN_ENCODED $OUT_ENCODED
#mkdir -p $OUTDIR

# Preprocess test set
echo "Preprocessing testset..."
python3 $TOOLS_DIR/spm_encode.py --model=$DATA_DIR/spm.model --infile $IN_FILE --outfile $IN_ENCODED/$IN_NAME.bpe

# Decode test set
CUDA_VISIBLE_DEVICES=$GPU sockeye-translate --models=$MODEL_DIR \
                  --input $IN_ENCODED/$IN_NAME.bpe \
                  --output $OUT_ENCODED/$OUT_NAME.bpe \
                  --batch-size 32 \
                  --max-output-length 144

# Postprocess test set
echo "Postprocessing testset..."
python3 $TOOLS_DIR/spm_decode.py --model=$DATA_DIR/spm.model --infile $OUT_ENCODED/$OUT_NAME.bpe --outfile $OUT_FILE

echo "Done translations for $IN_FILE and saved in $OUT_FILE."
