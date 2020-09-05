. path.sh

gpu=$1
label_type=char
DATA_DIR=data/ma/char
data_name=test
MODEL_PATH=exp/finetune_ma_char/checkpoint_best.pt
RESULT_DIR=exp/finetune_ma_char/decode_callhome_ma_beam1

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/speech_recognition/infer_ma.py $DATA_DIR \
--task audio_pretraining --nbest 1 --path $MODEL_PATH \
--gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder ctc_decoder \
--criterion ctc --labels $label_type --max-tokens 4000000 \
--post-process char --beam 1
