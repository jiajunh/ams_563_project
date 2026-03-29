import os
import random
import argparse
import json

import torch
import numpy as np
import matplotlib.pyplot as plt

from vae_model import VAE


def parse_args():
    parser = argparse.ArgumentParser()

    # Input / output
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to one .npz file for inference"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="infer_results",
        help="Directory to save inference outputs"
    )

    # Defaults aligned with training script
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--do_normalize", action="store_true")

    parser.add_argument("--base_channels", type=int, default=4, choices=[4, 8, 16])
    parser.add_argument("--latent_hw", type=int, default=128, choices=[128, 64])
    parser.add_argument("--first_kernel_size", type=int, default=5, choices=[3, 5])
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=24,
        choices=[24, 30, 36, 48],
        help="should be 3*k, k is int"
    )
    parser.add_argument("--shared_channels", type=int, default=64, choices=[128, 64])

    # Fixed by training script
    parser.add_argument("--in_channels", type=int, default=2)
    parser.add_argument("--out_channels", type=int, default=2)

    # Save options
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--save_npy", action="store_true")

    args = parser.parse_args()
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def maybe_normalize_volume(x):
    """
    x: [C, D, H, W]
    Normalize each channel independently over the whole volume.
    """
    x = x.astype(np.float32)
    out = np.zeros_like(x, dtype=np.float32)

    for c in range(x.shape[0]):
        mean = x[c].mean()
        std = x[c].std()
        if std < 1e-8:
            out[c] = x[c] - mean
        else:
            out[c] = (x[c] - mean) / std

    return out


def load_input(npz_path, do_normalize=False):
    data = np.load(npz_path)

    if "data" in data:
        x = data["data"]
    elif "image" in data:
        x = data["image"]
    else:
        raise KeyError(
            f"Neither 'data' nor 'image' found in {npz_path}. Keys: {list(data.keys())}"
        )

    print("Loaded keys:", list(data.keys()))
    print("Raw input shape:", x.shape)

    x = x.astype(np.float32)

    # Expected: [2, D, H, W]
    if x.ndim != 4:
        raise ValueError(
            f"Expected 3D volume [C, D, H, W], but got shape {x.shape}"
        )

    if x.shape[0] != 2:
        raise ValueError(
            f"Expected channel dimension = 2 (CT and PET), but got shape {x.shape}"
        )

    if do_normalize:
        x = maybe_normalize_volume(x)

    return x


def build_model(args, checkpoint_args=None):
    base_channels = args.base_channels
    shared_channels = args.shared_channels
    latent_dim = args.latent_dim
    latent_hw = args.latent_hw
    first_kernel_size = args.first_kernel_size
    in_channels = args.in_channels
    out_channels = args.out_channels

    if checkpoint_args is not None:
        base_channels = checkpoint_args.get("base_channels", base_channels)
        shared_channels = checkpoint_args.get("shared_channels", shared_channels)
        latent_dim = checkpoint_args.get("latent_dim", latent_dim)
        latent_hw = checkpoint_args.get("latent_hw", latent_hw)
        first_kernel_size = checkpoint_args.get("first_kernel_size", first_kernel_size)

    model = VAE(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        shared_channels=shared_channels,
        latent_dim=latent_dim,
        latent_hw=latent_hw,
        first_kernel_size=first_kernel_size,
    )

    return model


def run_model(model, x):
    with torch.no_grad():
        try:
            outputs = model(x, sample_posterior=False)
        except TypeError:
            outputs = model(x)

    if isinstance(outputs, (tuple, list)):
        if len(outputs) == 3:
            recon, mu, logvar = outputs
        elif len(outputs) == 2:
            recon, mu = outputs
            logvar = None
        elif len(outputs) == 1:
            recon = outputs[0]
            mu = None
            logvar = None
        else:
            raise ValueError(f"Unexpected number of outputs: {len(outputs)}")
    else:
        recon = outputs
        mu = None
        logvar = None

    return recon, mu, logvar


def tensor_to_numpy(x):
    if x is None:
        return None
    x = x.detach().cpu().numpy()
    if x.ndim > 0 and x.shape[0] == 1:
        x = np.squeeze(x, axis=0)
    return x


def save_mid_slice_visualization(input_np, recon_np, output_dir):
    """
    input_np, recon_np: [2, D, H, W]
    channel 0 = CT
    channel 1 = PET
    """
    os.makedirs(output_dir, exist_ok=True)

    mid_d = input_np.shape[1] // 2

    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(input_np[0, mid_d], cmap="gray")
    plt.title(f"Input CT (d={mid_d})")
    plt.axis("off")

    plt.subplot(2, 2, 2)
    plt.imshow(recon_np[0, mid_d], cmap="gray")
    plt.title(f"Recon CT (d={mid_d})")
    plt.axis("off")

    plt.subplot(2, 2, 3)
    plt.imshow(input_np[1, mid_d], cmap="gray")
    plt.title(f"Input PET (d={mid_d})")
    plt.axis("off")

    plt.subplot(2, 2, 4)
    plt.imshow(recon_np[1, mid_d], cmap="gray")
    plt.title(f"Recon PET (d={mid_d})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mid_slice_ct_pet.png"), dpi=200)
    plt.close()


if __name__ == "__main__":
    args = parse_args()
    print(args)

    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"'model_state_dict' not found in checkpoint: {args.checkpoint}. "
            f"Keys: {list(ckpt.keys())}"
        )

    checkpoint_args = ckpt.get("args", None)

    # Build model with training-time architecture if available
    model = build_model(args, checkpoint_args=checkpoint_args).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Prefer do_normalize stored in checkpoint
    do_normalize = args.do_normalize
    if checkpoint_args is not None and ("do_normalize" in checkpoint_args):
        do_normalize = checkpoint_args["do_normalize"] or do_normalize

    # Load one 3D two-channel volume: [2, D, H, W]
    input_np = load_input(args.input_path, do_normalize=do_normalize)

    # Add batch dimension for Conv3d VAE: [1, 2, D, H, W]
    x = torch.from_numpy(input_np).unsqueeze(0).to(device)

    # Whole-volume inference
    recon, mu, logvar = run_model(model, x)

    # Remove batch dimension from reconstruction
    recon_np = recon.squeeze(0).detach().cpu().numpy()  # [2, D, H, W]

    # Split channels
    input_ct = input_np[0]    # [D, H, W]
    input_pet = input_np[1]   # [D, H, W]
    recon_ct = recon_np[0]    # [D, H, W]
    recon_pet = recon_np[1]   # [D, H, W]

    mu_np = tensor_to_numpy(mu)
    logvar_np = tensor_to_numpy(logvar)

    # Save combined and split outputs
    save_dict = {
        "input": input_np,        # [2, D, H, W]
        "recon": recon_np,        # [2, D, H, W]
        "input_ct": input_ct,     # [D, H, W]
        "input_pet": input_pet,   # [D, H, W]
        "recon_ct": recon_ct,     # [D, H, W]
        "recon_pet": recon_pet,   # [D, H, W]
    }

    if mu_np is not None:
        save_dict["mu"] = mu_np
    if logvar_np is not None:
        save_dict["logvar"] = logvar_np

    np.savez(os.path.join(args.output_dir, "vae_infer_result.npz"), **save_dict)

    # Optional .npy outputs
    if args.save_npy:
        np.save(os.path.join(args.output_dir, "input_2ch.npy"), input_np)
        np.save(os.path.join(args.output_dir, "recon_2ch.npy"), recon_np)

        np.save(os.path.join(args.output_dir, "input_ct.npy"), input_ct)
        np.save(os.path.join(args.output_dir, "input_pet.npy"), input_pet)
        np.save(os.path.join(args.output_dir, "recon_ct.npy"), recon_ct)
        np.save(os.path.join(args.output_dir, "recon_pet.npy"), recon_pet)

        if mu_np is not None:
            np.save(os.path.join(args.output_dir, "mu.npy"), mu_np)
        if logvar_np is not None:
            np.save(os.path.join(args.output_dir, "logvar.npy"), logvar_np)

    # Optional PNG visualization
    if args.save_png:
        save_mid_slice_visualization(input_np, recon_np, args.output_dir)

    # Save metadata
    meta = {
        "checkpoint": args.checkpoint,
        "input_path": args.input_path,
        "device": str(device),
        "input_shape": list(input_np.shape),
        "recon_shape": list(recon_np.shape),
        "ct_shape": list(input_ct.shape),
        "pet_shape": list(input_pet.shape),
        "checkpoint_epoch": ckpt.get("epoch", None),
        "best_loss": ckpt.get("best_loss", None),
        "used_do_normalize": do_normalize,
        "model_args_from_checkpoint": str(checkpoint_args),
        "channel_definition": {
            "0": "CT",
            "1": "PET"
        }
    }

    with open(os.path.join(args.output_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("Inference finished.")
    print("Input shape      :", input_np.shape)
    print("Recon shape      :", recon_np.shape)
    print("Input CT shape   :", input_ct.shape)
    print("Input PET shape  :", input_pet.shape)
    print("Recon CT shape   :", recon_ct.shape)
    print("Recon PET shape  :", recon_pet.shape)

    if mu_np is not None:
        print("Mu shape         :", mu_np.shape)
    if logvar_np is not None:
        print("Logvar shape     :", logvar_np.shape)

    print("Results saved to :", args.output_dir)