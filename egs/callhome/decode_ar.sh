. ../path.sh

gpu=$1
label_type=subword
DATA_DIR=data/ar
data_name=test
MODEL_PATH=exp/finetune_6lang_subword/checkpoint_best.pt
RESULT_DIR=exp/finetune_6lang_subword/decode_callhome_ar_beam1

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/speech_recognition/infer.py $DATA_DIR \
--task audio_pretraining --nbest 1 --path $MODEL_PATH \
--gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder ctc_decoder \
--criterion ctc --labels $label_type --max-tokens 4000000 \
--post-process $label_type --beam 1


# . ./path.sh
#
# gpu=$1
# lm_weight=1.46
# word_score=0.52
# label_type=char
# DATA_DIR=data/ja/char
# data_name=test
# lexicon=data/6lang_subword/lexicon.txt
# lm=data/6lang_subword/5-gram.bin
# MODEL_PATH=exp/finetune_6lang_subword_3/checkpoint_best.pt
# RESULT_DIR=exp/finetune_6lang_subword_3/decode_callhome_ja_selflm${lm_weight}_score${word_score}
#
# CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/speech_recognition/infer.py $DATA_DIR \
# --task audio_pretraining --nbest 1 --path $MODEL_PATH \
# --gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder kenlm \
# --lm-model $lm --lm-weight ${lm_weight} \
# --word-score ${word_score} --sil-weight 0 --criterion ctc --labels $label_type --max-tokens 4000000 \
# --post-process letter --lexicon $lexicon --beam 100
