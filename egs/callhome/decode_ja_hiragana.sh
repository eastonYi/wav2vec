. path.sh

gpu=0
lm_weight=1.46
word_score=0.52
label_type=hga
DATA_DIR=data/hiragana
data_name=test
lexicon=data/hiragana/lexicon.txt
lm=data/callhome.word.bin
# MODEL_PATH=exp/finetune_ja_hiragana/checkpoint_best.pt
MODEL_PATH=/data5/syzhou/work/data/fairseq/exp/wav2vec2_base_finetune_callhome_ja/checkpoint_best.pt
RESULT_DIR=exp/finetune_ja_hiragana/decode_callhome_ja_selflm${lm_weight}_score${word_score}

CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/speech_recognition/infer_ma.py $DATA_DIR \
--task audio_pretraining --nbest 1 --path $MODEL_PATH \
--gen-subset $data_name --results-path $RESULT_DIR --w2l-decoder kenlm \
--lm-model $lm --lm-weight ${lm_weight} \
--word-score ${word_score} --sil-weight 0 --criterion ctc --labels $label_type --max-tokens 4000000 \
--post-process letter --lexicon $lexicon --beam 100
