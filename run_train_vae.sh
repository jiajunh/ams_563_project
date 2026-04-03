#!/bin/bash


python train_vae.py \
    --train_data_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train" \
    --test_data_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train" \
    --data_augment \
    --num_workers 1 \
    --test_num_workers 4 \
    --seed 1 \
    --batch_size 1 \
    --test_batch_size 4 \
    --lr 1.0e-4 \
    --min_lr_ratio 0.1 \
    --lr_scheduler "cosine" \
    --warmup_frac 0.1 \
    --epochs 30 \
    --evaluate_freq 1 \
    --test_chunk_size 64 \
    --weight_decay 1.0e-5 \
    --grad_clip 1.0 \
    --kl_coef 1.0e-4 \
    --kl_warmup_frac 0.25 \
    --vae_choice "large" \
    --base_channels 8 \
    --latent_dim 18 \
    --shared_channels 64 \
    --perception_loss_slices 16 \
    --perception_chunk_size 32 \
    --perceptual_weight 0.1 \
    --independent_weight 3.0 \

