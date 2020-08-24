GPUS=0,1
DATA_DIR=data/uyghur/train
SAVE_DIR=exp/wav2vec2_base_finetune_ar
MODEL_PATH=exp/wav2vec2_base_pretrain/checkpoint_best.pt

. path.sh

CUDA_VISIBLE_DEVICES=$GPUS python train.py --save-dir $SAVE_DIR --fp16 \
--wer-args '("/path/to/lm/4-gram.bin","/path/to/lexicon",2,-1)' \
--post-process letter --train-subset train  --valid-subset valid --no-epoch-checkpoints --best-checkpoint-metric wer --num-workers 4 \
--max-update 80000 --sentence-avg --task audio_pretraining --arch wav2vec_ctc --w2v-path $MODEL_PATH \
--labels ltr --apply-mask --mask-selection static --mask-other 0 --mask-length 10 --mask-prob 0.5 --layerdrop 0.1 \
--mask-channel-selection static --mask-channel-other 0 --mask-channel-length 64 --mask-channel-prob 0.5 --zero-infinity \
--feature-grad-mult 0.0 --freeze-finetune-updates 10000 --validate-after-updates 10000 --optimizer adam \
--adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --lr 2e-05 --lr-scheduler tri_stage --warmup-steps 8000 --hold-steps 32000 \
--decay-steps 40000 --final-lr-scale 0.05 --final-dropout 0.0 --dropout 0.0 --activation-dropout 0.1 --criterion ctc \
--attention-dropout 0.0 --max-tokens 1280000 --seed 2337 --log-format simple --log-interval 50 --ddp-backend no_c10d \
--save-interval 50
