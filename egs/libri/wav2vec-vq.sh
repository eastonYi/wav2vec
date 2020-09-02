DATA_DIR=data
OUTPUT_DIR=data/vq
MODEL_PATH=exp/vq-wav2vec.pt

. path.sh

CUDA_VISIBLE_DEVICES=0 python $SRC_ROOT/vq-wav2vec_featurize.py --data-dir $DATA_DIR --output-dir $OUTPUT_DIR \
--checkpoint $MODEL_PATH --split dev-clean dev-other --extension tsv --log-format tqdm
