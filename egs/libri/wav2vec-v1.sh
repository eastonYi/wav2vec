DATA_DIR=/data5/syzhou/work/data/corpus/LibriSpeech
OUTPUT_DIR=data/dev-clean_vec
MODEL_PATH=exp/wav2vec_large.pt

. path.sh

CUDA_VISIBLE_DEVICES=0 python $SRC_ROOT/wav2vec_featurize.py --input $DATA_DIR --output $OUTPUT_DIR \
--model $MODEL_PATH --ext flac --split dev-clean --no-copy-labels
