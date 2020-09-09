RESULT_DIR=$1
NAME=$2
/home/easton/files/sctk-2.4.10/bin/sclite -r ${RESULT_DIR}/ref.word-checkpoint_best.pt-${NAME}.txt trn -h ${RESULT_DIR}/hypo.word-checkpoint_best.pt-${NAME}.txt -i rm -c NOASCII -s -o all stdout > ${RESULT_DIR}/${NAME}.result.wrd.txt
vi ${RESULT_DIR}/${NAME}.result.wrd.txt
