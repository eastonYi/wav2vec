DATA_DIR=data/org_wav
OUTPUT_DIR=data/org_vec
MODEL_PATH=exp/wav2vec_large.pt

. path.sh

python $SRC_ROOT/wav2vec_featurize.py --input $DATA_DIR --output $OUTPUT_DIR \
--model $MODEL_PATH --split train
