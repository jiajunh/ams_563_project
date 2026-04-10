#!/bin/bash


python train_segmentation.py \
    --train_image_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/train/images/" \
    --train_label_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/train/labels/" \
    --test_image_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/test/images/" \
    --test_label_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/samples/test/labels/" \
    --log_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/ams_563_project/logs/" \
    --vae_ckpt_path "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/checkpoints/epoch_030.pt" \
    --ckpt_dir "/n/netscratch/kdbrantley_lab/Lab/jiajunh/autopet/checkpoints/decoder/" \
    --data_augment \
    --num_workers 1 \
    --test_num_workers 4 \
    --seed 1 \
    --batch_size 2 \
    --test_batch_size 4 \
    --lr 10.0e-4 \
    --min_lr_ratio 0.1 \
    --lr_scheduler "cosine" \
    --warmup_frac 0.1 \
    --epochs 30 \
    --evaluate_freq 1 \
    --weight_decay 1e-4 \
    --grad_clip 1.0 \
    --dino_multi_scale 3 \
    --focal_gamma 2.0 \
    --focal_alpha 0.95 \
    --dice_weight 1.0 \
    --focal_weight 50.0 \
    # --use_dino \



