RESULT_DIR=exp/finetune_ja_hiragana/decode_callhome_ja_selflm1.46_score0.52
NAME=test
sclite -r ${RESULT_DIR}/ref.word-checkpoint_best.pt-${NAME}.txt trn -h ${RESULT_DIR}/hypo.word-checkpoint_best.pt-${NAME}.txt -i rm -c NOASCII -s -o all stdout > ${RESULT_DIR}/${NAME}.result.wrd.txt
vi ${RESULT_DIR}/${NAME}.result.wrd.txt
