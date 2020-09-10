. path.sh

gpu=$1
SAVE_DIR=exp/finetune_ma_char
W2V_PATH=exp/pretrain_6lang/checkpoint_best.pt
DATA_DIR=data/ma/char
label_type=char

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python $SRC_ROOT/train.py $DATA_DIR \
--save-dir $SAVE_DIR --tensorboard-logdir $SAVE_DIR --post-process $label_type \
--train-subset train --valid-subset valid --no-epoch-checkpoints --best-checkpoint-metric uer \
--num-workers 4 --max-update 80000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc \
--w2v-path $W2V_PATH --labels $label_type --apply-mask --mask-selection static --mask-other 0 \
--mask-length 10 --mask-prob 0.5 --layerdrop 0.1 --mask-channel-selection static \
--mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 \
--validate-interval 50 --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 \
--lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 42000 --decay-steps 50000 \
--final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
--attention-dropout 0.0 --max-tokens 1000000 --seed 2337 --ddp-backend no_c10d \
--update-freq 12 --log-interval 10 --log-format simple --save-interval 50
