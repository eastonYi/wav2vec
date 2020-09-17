. ./path.sh

gpu=$1
SAVE_DIR=../exp/finetune_seq2seq_ge_subword
W2V_PATH=exp/pretrain_6lang/checkpoint_best.pt
DATA_DIR=../data/ge/subword
label_type=subword

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/train.py $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --post-process $label_type \
--train-subset train --valid-subset dev_ge --no-epoch-checkpoints --best-checkpoint-metric loss \
--num-workers 2 --max-update 80000 \
--task audio_pretraining --arch wav2vec_seq2seq --w2v-path $W2V_PATH --labels $label_type \
--decoder-embed-dim 768 --decoder-ffn-embed-dim 3072 --decoder-layers 4 --decoder-layerdrop 0.0 \
--decoder-attention-heads 8 --decoder-learned-pos --decoder-normalize-before --decoder-dropout 0.1 \
--decoder-attention-dropout 0.1 --decoder-activation-dropout 0.1 \
--apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 \
--feature-grad-mult 0.0 --freeze-finetune-updates 3000 --validate-after-updates 3000  --validate-interval 2000 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage \
--warmup-steps 5000 --hold-steps 20000 --decay-steps 30000 --final-lr-scale 0.05 \
--final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion cross_entropy_uer \
--attention-dropout 0.0 --max-tokens 800000 --seed 2337 --ddp-backend no_c10d --update-freq 6 \
--log-interval 20 --log-format simple --save-interval 20
