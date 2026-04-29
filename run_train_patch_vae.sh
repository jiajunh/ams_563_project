#!/bin/bash

python train_patch_vae.py \
    --train_data_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train" \
    --test_data_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/data/for_vae/train" \
    --num_workers 1 \
    --test_num_workers 4 \
    --seed 1 \
    --batch_size 1 \
    --minibatch_size 16 \
    --test_batch_size 1 \
    --lr 2.0e-4 \
    --min_lr_ratio 0.1 \
    --lr_scheduler "cosine" \
    --warmup_frac 0.1 \
    --epochs 50 \
    --evaluate_freq 2 \
    --test_chunk_size 128 \
    --weight_decay 1.0e-4 \
    --grad_clip 1.0 \
    --kl_coef 1.0e-5 \
    --kl_warmup_frac 0.20 \
    --base_channels 16 \
    --d_base_channels 16 \
    --latent_dim 16 \
    --patch_size 64 \
    --stride 48 \
    --perception_loss_slices 4 \
    --perception_chunk_size 16 \
    --perceptual_weight 0.1 \
    --g_loss_weight 0.2 \
    --use_gan \



