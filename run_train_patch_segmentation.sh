#!/bin/bash

python train_patch_segmentation.py \
    --train_data_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/train/images/" \
    --train_label_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/train/labels/" \
    --test_data_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/test/images/" \
    --test_label_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/test/labels/" \
    --num_workers 1 \
    --test_num_workers 4 \
    --seed 1 \
    --dataset_type "balanced" \
    --sample_patches_per_volume 64 \
    --sample_positive_ratio 0.5 \
    --sample_pos_jitter 20 \
    --batch_size 4 \
    --minibatch_size 32 \
    --test_batch_size 1 \
    --lr 2.0e-4 \
    --min_lr_ratio 0.1 \
    --lr_scheduler "cosine" \
    --warmup_frac 0.1 \
    --epochs 100 \
    --evaluate_freq 1 \
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
    --focal_gamma 2.0 \
    --focal_alpha 0.6 \
    --dice_weight 1.0 \
    --focal_weight 2.0 \
    --g_loss_weight 0.1 \
    --use_gan \


# python train_patch_segmentation.py \
#     --train_data_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/train/images/" \
#     --train_label_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/train/labels/" \
#     --test_data_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/test/images/" \
#     --test_label_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/test/labels/" \
#     --num_workers 1 \
#     --test_num_workers 4 \
#     --seed 1 \
#     --dataset_type "all" \
#     --sample_patches_per_volume 64 \
#     --sample_positive_ratio 0.5 \
#     --sample_pos_jitter 20 \
#     --batch_size 1 \
#     --minibatch_size 32 \
#     --test_batch_size 1 \
#     --lr 2.0e-4 \
#     --min_lr_ratio 0.1 \
#     --lr_scheduler "cosine" \
#     --warmup_frac 0.1 \
#     --epochs 50 \
#     --evaluate_freq 1 \
#     --test_chunk_size 128 \
#     --weight_decay 1.0e-4 \
#     --grad_clip 1.0 \
#     --kl_coef 1.0e-5 \
#     --kl_warmup_frac 0.20 \
#     --base_channels 16 \
#     --d_base_channels 16 \
#     --latent_dim 16 \
#     --patch_size 64 \
#     --stride 48 \
#     --focal_gamma 2.0 \
#     --focal_alpha 0.95 \
#     --dice_weight 1.0 \
#     --focal_weight 20.0 \
#     --g_loss_weight 0.1 \
#     --use_gan \


