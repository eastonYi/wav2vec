. ../path.sh

gpu=$1
label_type=char
DATA_DIR=data/ja/char
data_name=test
MODEL_PATH=exp/finetune_ja_char/checkpoint_best.pt
RESULT_DIR=exp/finetune_ja_char/decode_callhome_ja_beam1

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/speech_recognition/infer.py \
$DATA_DIR --task audio_pretraining --nbest 1 --path $MODEL_PATH \
--gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder ctc_decoder \
--criterion ctc --labels $label_type --iscn --max-tokens 4000000 \
--post-process $label_type --beam 1
