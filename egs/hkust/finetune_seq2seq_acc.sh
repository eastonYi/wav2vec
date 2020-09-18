. ../path.sh

gpu=$1
SAVE_DIR=exp/finetune_seq2seq_acc
W2V_PATH=../libri/wav2vec2_small.pt
DATA_DIR=data
label_type=char

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/train.py $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --post-process $label_type \
--train-subset train --valid-subset valid --no-epoch-checkpoints --best-checkpoint-metric acc \
--num-workers 2 --max-update 80000 --task audio_pretraining --arch wav2vec_seq2seq --w2v-path $W2V_PATH \
--decoder-embed-dim 768 --decoder-ffn-embed-dim 3072 --decoder-layers 6 --decoder-layerdrop 0.0 \
--decoder-learned-pos --decoder-normalize-before \
--decoder-dropout 0.1 --decoder-attention-dropout 0.1 --decoder-activation-dropout 0.1 \
--labels $label_type --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 \
--feature-grad-mult 0.0 --freeze-finetune-updates 1 --validate-after-updates 1  --validate-interval 10 \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 8e-05 --lr-scheduler tri_stage \
--warmup-steps 5000 --hold-steps 20000 --decay-steps 30000 --final-lr-scale 0.05 \
--final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion cross_entropy_acc \
--attention-dropout 0.0 --max-tokens 800000 --seed 2337 --ddp-backend no_c10d --update-freq 12 \
--log-interval 50 --log-format simple --save-interval 10
