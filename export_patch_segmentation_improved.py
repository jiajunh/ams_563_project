import os
import csv
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

try:
    import nibabel as nib
except ImportError:
    nib = None

try:
    from scipy import ndimage as ndi
except ImportError:
    ndi = None

from patch_vae_model import PatchSegmentation3D


def parse_args():
    parser = argparse.ArgumentParser(description="Export PatchVAE segmentation results in original-volume space.")

    parser.add_argument("--test_data_path", type=str, required=True)
    parser.add_argument("--test_label_path", type=str, default="")
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=None)
    parser.add_argument("--test_chunk_size", type=int, default=None)
    parser.add_argument("--test_batch_size", type=int, default=1)
    parser.add_argument("--test_num_workers", type=int, default=4)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--ct_channel", type=int, default=0)
    parser.add_argument("--pet_channel", type=int, default=1)
    parser.add_argument("--overlay_alpha", type=float, default=0.35)

    parser.add_argument("--target_size", type=int, default=224)
    parser.add_argument("--save_nii", action="store_true")
    parser.add_argument("--save_npz", action="store_true")
    parser.add_argument("--save_png", action="store_true")
    parser.add_argument("--save_prob_nii", action="store_true")
    parser.add_argument("--save_target_npz", action="store_true",
                        help="Also save the 224-space image/prob/pred used internally by the network.")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    parser.add_argument("--lesion_bins_ml", type=str, default="0,1,5,10,50,inf",
                        help="Comma-separated lesion volume bins in mL, e.g. 0,1,5,10,50,inf")
    parser.add_argument("--lesion_min_overlap_voxels", type=int, default=1,
                        help="Minimum overlapping voxels required to call a GT lesion detected.")
    parser.add_argument("--lesion_min_overlap_ratio", type=float, default=0.0,
                        help="Minimum overlap ratio relative to the GT lesion volume required for detection.")

    return parser.parse_args()


def copy_args_from_checkpoint(args, ckpt):
    """Use training arguments saved in the checkpoint unless they are overridden."""
    ckpt_args = ckpt.get("args", {})

    for name, default_value in [
        ("patch_size", 64),
        ("stride", 48),
        ("base_channels", 16),
        ("latent_dim", 16),
        ("test_chunk_size", 128),
    ]:
        if getattr(args, name) is None:
            setattr(args, name, ckpt_args.get(name, default_value))

    return args


def get_case_name(file_name):
    """Remove common medical image/data suffixes while preserving the original case name."""
    name = Path(file_name).name
    for suffix in [".nii.gz", ".nii", ".npz", ".npy", ".mha", ".mhd", ".nrrd"]:
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return Path(name).stem


def normalize_image_shape(image):
    """Return image as [C, D, H, W]."""
    if image.ndim == 3:
        image = image[None, ...]
    if image.ndim != 4:
        raise ValueError(f"Expected image with shape [C,D,H,W] or [D,H,W], got {image.shape}")
    return image


def normalize_label_shape(label):
    """Return label as [D, H, W]."""
    if label is None:
        return None

    if label.ndim == 4:
        if label.shape[0] == 1:
            label = label[0]
        else:
            label = label[0]

    if label.ndim != 3:
        raise ValueError(f"Expected label with shape [D,H,W] or [1,D,H,W], got {label.shape}")

    return label


def resize_image_and_label(x, y, target_shape):
    """Resize x [C,D,H,W] and y [D,H,W] to target_shape for network inference."""
    if tuple(x.shape[1:]) != tuple(target_shape):
        x = x.unsqueeze(0)
        x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
        x = x.squeeze(0)

    if y is not None and tuple(y.shape) != tuple(target_shape):
        y = y.unsqueeze(0).unsqueeze(0).float()
        y = F.interpolate(y, size=target_shape, mode="nearest")
        y = y.squeeze(0).squeeze(0)

    return x, y


class ExportFullImagePatchDataset3D(Dataset):
    """Full-volume patch dataset for inference and original-space export."""
    def __init__(self, image_dir, label_dir=None, patch_size=64, stride=48, target_size=224):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.patch_size = patch_size
        self.stride = stride
        self.target_shape = (target_size, target_size, target_size)
        self.dtype = torch.float32

        self.files = sorted([
            f for f in os.listdir(image_dir)
            if f.endswith(".npz")
        ])

    def __len__(self):
        return len(self.files)

    def _get_starts(self, size):
        P = self.patch_size
        S = self.stride
        starts = list(range(0, size - P + 1, S))
        if len(starts) == 0 or starts[-1] != size - P:
            starts.append(size - P)
        return starts

    def _compute_coord(self, sd, sh, sw, D, H, W):
        half = self.patch_size / 2.0
        return [(sd + half) / D, (sh + half) / H, (sw + half) / W]

    def __getitem__(self, idx):
        file_name = self.files[idx]
        case_name = get_case_name(file_name)
        img_path = os.path.join(self.image_dir, file_name)

        img_data = np.load(img_path)
        img_orig_np = normalize_image_shape(img_data["image"]).astype(np.float32)
        original_shape = tuple(img_orig_np.shape[-3:])

        label_orig_np = None
        if self.label_dir is not None:
            label_path = os.path.join(self.label_dir, file_name)
            if os.path.exists(label_path):
                label_data = np.load(label_path)
                label_orig_np = normalize_label_shape(label_data["label"]).astype(np.float32)

        x_orig = torch.from_numpy(img_orig_np).to(self.dtype)
        y_orig = torch.from_numpy(label_orig_np).to(self.dtype) if label_orig_np is not None else None

        x = x_orig.clone()
        y = y_orig.clone() if y_orig is not None else None
        x, y = resize_image_and_label(x, y, self.target_shape)

        C, D, H, W = x.shape
        P = self.patch_size
        d_starts = self._get_starts(D)
        h_starts = self._get_starts(H)
        w_starts = self._get_starts(W)

        patches = []
        coords = []
        starts = []

        for sd in d_starts:
            for sh in h_starts:
                for sw in w_starts:
                    patches.append(x[:, sd:sd + P, sh:sh + P, sw:sw + P])
                    coords.append(self._compute_coord(sd, sh, sw, D, H, W))
                    starts.append((sd, sh, sw))

        out = {
            "image": x,
            "image_orig": x_orig,
            "patch": torch.stack(patches, dim=0),
            "coord": torch.tensor(coords, dtype=torch.float32),
            "start": torch.tensor(starts, dtype=torch.long),
            "case_name": case_name,
            "file_name": file_name,
            "image_path": img_path,
            "original_shape": torch.tensor(original_shape, dtype=torch.long),
        }

        if y is not None:
            out["label"] = y.unsqueeze(0)
        if y_orig is not None:
            out["label_orig"] = y_orig.unsqueeze(0)

        return out


def load_affine_from_npz(npz_path):
    """Load affine if it exists in the NPZ file; otherwise build one from spacing."""
    try:
        data = np.load(npz_path)
        if "affine" in data:
            return data["affine"].astype(np.float32)
        if "spacing" in data:
            spacing = np.asarray(data["spacing"], dtype=np.float32).reshape(-1)
            affine = np.eye(4, dtype=np.float32)
            if spacing.size >= 3:
                affine[0, 0] = spacing[0]
                affine[1, 1] = spacing[1]
                affine[2, 2] = spacing[2]
            return affine
    except Exception:
        pass
    return np.eye(4, dtype=np.float32)


def load_geometry_from_npz(npz_path, strict=False):
    """
    Load geometry information from NPZ.

    Priority:
        1. top-level affine
        2. top-level spacing
        3. properties["spacing_ct"]
        4. properties["spacing_pet"]

    Note:
        spacing is stored in xyz order.
        NumPy array is usually zyx order.
        For volume calculation, order does not matter.
    """
    data = np.load(npz_path, allow_pickle=True)

    if "affine" in data:
        affine = data["affine"].astype(np.float32)
        spacing_xyz = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)).astype(np.float32)
        return affine, spacing_xyz, "affine"

    if "spacing" in data:
        spacing_xyz = np.asarray(data["spacing"], dtype=np.float32).reshape(-1)[:3]

        affine = np.eye(4, dtype=np.float32)
        affine[0, 0] = spacing_xyz[0]
        affine[1, 1] = spacing_xyz[1]
        affine[2, 2] = spacing_xyz[2]

        return affine, spacing_xyz, "spacing"

    if "properties" in data:
        props = data["properties"].item()

        if "spacing_ct" in props:
            spacing_xyz = np.asarray(props["spacing_ct"], dtype=np.float32).reshape(-1)[:3]
            source = "properties.spacing_ct"
        elif "spacing_pet" in props:
            spacing_xyz = np.asarray(props["spacing_pet"], dtype=np.float32).reshape(-1)[:3]
            source = "properties.spacing_pet"
        else:
            spacing_xyz = None
            source = "properties.no_spacing"

        if spacing_xyz is not None:
            affine = np.eye(4, dtype=np.float32)
            affine[0, 0] = spacing_xyz[0]
            affine[1, 1] = spacing_xyz[1]
            affine[2, 2] = spacing_xyz[2]

            return affine, spacing_xyz, source

    msg = f"[SpacingCheck] No affine, spacing, or properties spacing found in {npz_path}."
    if strict:
        raise ValueError(msg)

    print("WARNING:", msg, "Using 1x1x1 mm spacing.")
    affine = np.eye(4, dtype=np.float32)
    spacing_xyz = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    return affine, spacing_xyz, "missing"


def load_geometry_from_npz1(npz_path, strict=False):
    """
    Load affine and spacing information from NPZ.

    Returns:
        affine: 4x4 affine matrix
        spacing: spacing inferred from affine or stored spacing
        spacing_source: "affine", "spacing", or "missing"
    """
    data = np.load(npz_path)

    if "affine" in data:
        affine = data["affine"].astype(np.float32)

        if affine.shape != (4, 4):
            msg = f"[SpacingCheck] Invalid affine shape in {npz_path}: {affine.shape}"
            if strict:
                raise ValueError(msg)
            print("WARNING:", msg)

        spacing = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0)).astype(np.float32)
        return affine, spacing, "affine"

    if "spacing" in data:
        spacing = np.asarray(data["spacing"], dtype=np.float32).reshape(-1)

        if spacing.size < 3:
            msg = f"[SpacingCheck] Invalid spacing in {npz_path}: {spacing}"
            if strict:
                raise ValueError(msg)
            print("WARNING:", msg)
            spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        spacing = spacing[:3]

        affine = np.eye(4, dtype=np.float32)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]

        return affine, spacing, "spacing"

    msg = f"[SpacingCheck] No affine or spacing found in {npz_path}. Using 1x1x1 mm spacing."
    if strict:
        raise ValueError(msg)

    print("WARNING:", msg)

    affine = np.eye(4, dtype=np.float32)
    spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    return affine, spacing, "missing"


def voxel_volume_ml_from_affine(affine):
    """Compute voxel volume in mL from a 4x4 affine. Assumes affine units are mm."""
    try:
        voxel_volume_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
        if not np.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
            return 0.001
        return voxel_volume_mm3 / 1000.0
    except Exception:
        return 0.001


def check_spacing_info(case_name, spacing, affine, voxel_volume_ml, original_shape):
    """
    Print spacing and voxel volume information for debugging.
    """
    spacing = np.asarray(spacing, dtype=np.float32).reshape(-1)

    spacing_product_ml = float(np.prod(spacing[:3]) / 1000.0)
    affine_det_ml = voxel_volume_ml

    print(
        f"[SpacingCheck] {case_name} | "
        f"shape={tuple(original_shape)} | "
        f"spacing={spacing[:3].tolist()} mm | "
        f"voxel_volume_from_spacing={spacing_product_ml:.8f} mL | "
        f"voxel_volume_from_affine={affine_det_ml:.8f} mL"
    )

    if not np.all(np.isfinite(spacing[:3])):
        raise ValueError(f"[SpacingCheck] Non-finite spacing for {case_name}: {spacing}")

    if np.any(spacing[:3] <= 0):
        raise ValueError(f"[SpacingCheck] Non-positive spacing for {case_name}: {spacing}")

    if voxel_volume_ml <= 0 or not np.isfinite(voxel_volume_ml):
        raise ValueError(f"[SpacingCheck] Invalid voxel volume for {case_name}: {voxel_volume_ml}")

    if not np.isclose(spacing_product_ml, affine_det_ml, rtol=1e-3, atol=1e-8):
        print(
            f"WARNING: [SpacingCheck] {case_name} spacing product and affine determinant differ: "
            f"{spacing_product_ml:.8f} vs {affine_det_ml:.8f} mL"
        )

    if np.allclose(spacing[:3], [1.0, 1.0, 1.0]):
        print(
            f"WARNING: [SpacingCheck] {case_name} spacing is [1, 1, 1]. "
            f"Check whether this is real original spacing or a missing metadata fallback."
        )


def build_model(args, ckpt, device):
    model = PatchSegmentation3D(
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels,
        latent_dim=args.latent_dim,
    ).to(device)

    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def infer_one_volume(args, model, batch, device):
    """Aggregate overlapping patch logits into one 224-space full-volume prediction."""
    x_full = batch["image"].to(device, non_blocking=True)
    patches = batch["patch"].to(device, non_blocking=True)
    coords = batch["coord"].to(device, non_blocking=True)
    starts = batch["start"].to(device, non_blocking=True)

    B, N, C, _, _, _ = patches.shape
    if B != 1:
        raise ValueError("Please set --test_batch_size 1 for full-volume export.")

    x_full = x_full[0]
    patches = patches[0]
    coords = coords[0]
    starts = starts[0]

    _, D, H, W = x_full.shape
    P = args.patch_size

    logit_full = torch.zeros((1, D, H, W), device=device)
    weight_full = torch.zeros((1, D, H, W), device=device)

    for start_idx in range(0, N, args.test_chunk_size):
        end_idx = min(start_idx + args.test_chunk_size, N)

        patch_batch = patches[start_idx:end_idx]
        coord_batch = coords[start_idx:end_idx]
        start_batch = starts[start_idx:end_idx]

        logits = model(patch_batch, coord_batch)

        for j in range(logits.shape[0]):
            sd, sh, sw = start_batch[j].tolist()
            logit_full[:, sd:sd + P, sh:sh + P, sw:sw + P] += logits[j]
            weight_full[:, sd:sd + P, sh:sh + P, sw:sw + P] += 1.0

    logit_full = logit_full / weight_full.clamp_min(1.0)
    prob_full = torch.sigmoid(logit_full)
    pred_full = (prob_full > args.threshold).float()

    return x_full, logit_full, prob_full, pred_full


def get_label_orig_from_batch(batch):
    if "label_orig" not in batch:
        return None
    y = batch["label_orig"][0].float()
    if y.ndim == 3:
        y = y.unsqueeze(0)
    return y


def resize_volume_np(volume, target_shape, mode):
    """Resize a [D, H, W] NumPy volume."""
    tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
    if mode == "nearest":
        out = F.interpolate(tensor, size=target_shape, mode="nearest")
    else:
        out = F.interpolate(tensor, size=target_shape, mode="trilinear", align_corners=False)
    return out.squeeze(0).squeeze(0).cpu().numpy()


def dice_np(pred, target, eps=1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    inter = np.logical_and(pred, target).sum()
    return float((2.0 * inter + eps) / (denom + eps))


def voxel_error_stats_ml(pred, target, voxel_volume_ml):
    """Return FP/FN voxel counts and FPV/FNV in mL."""
    pred_b = pred.astype(bool)
    target_b = target.astype(bool)
    fp = int(np.logical_and(pred_b, np.logical_not(target_b)).sum())
    fn = int(np.logical_and(np.logical_not(pred_b), target_b).sum())
    fpv_ml = fp * voxel_volume_ml
    fnv_ml = fn * voxel_volume_ml
    return fp, fn, float(fpv_ml), float(fnv_ml)


def parse_bins_ml(bin_text):
    bins = []
    for item in bin_text.split(","):
        item = item.strip().lower()
        if item in ["inf", "+inf", "infinity"]:
            bins.append(np.inf)
        else:
            bins.append(float(item))
    if len(bins) < 2:
        raise ValueError("At least two lesion volume bin edges are required.")
    if any(bins[i] >= bins[i + 1] for i in range(len(bins) - 1)):
        raise ValueError(f"Lesion bins must be strictly increasing, got {bins}")
    return bins


def bin_label(value, bins, idx):
    left = bins[idx]
    right = bins[idx + 1]
    if np.isinf(right):
        return f">={left:g}"
    return f"[{left:g},{right:g})"


def compute_lesion_detection(case_name, pred, target, voxel_volume_ml, args):
    """Compute GT-lesion detection and FP lesion statistics in original-volume space."""
    if ndi is None:
        raise ImportError("scipy is required for lesion-level connected-component analysis. Install it with: pip install scipy")

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    gt_cc, num_gt = ndi.label(target.astype(bool), structure=structure)
    pred_cc, num_pred = ndi.label(pred.astype(bool), structure=structure)

    lesion_rows = []
    detected_lesions = 0
    missed_lesions = 0

    for lesion_id in range(1, num_gt + 1):
        gt_mask = gt_cc == lesion_id
        lesion_voxels = int(gt_mask.sum())
        lesion_volume_ml = lesion_voxels * voxel_volume_ml
        overlap_voxels = int(np.logical_and(gt_mask, pred > 0).sum())
        overlap_ratio = float(overlap_voxels / max(lesion_voxels, 1))
        detected = (
            overlap_voxels >= args.lesion_min_overlap_voxels
            and overlap_ratio >= args.lesion_min_overlap_ratio
        )
        if detected:
            detected_lesions += 1
        else:
            missed_lesions += 1

        lesion_rows.append({
            "case_name": case_name,
            "gt_lesion_id": lesion_id,
            "gt_lesion_voxels": lesion_voxels,
            "gt_lesion_volume_ml": lesion_volume_ml,
            "overlap_voxels": overlap_voxels,
            "overlap_ratio": overlap_ratio,
            "detected": int(detected),
        })

    fp_lesion_rows = []
    false_positive_lesions = 0
    for pred_id in range(1, num_pred + 1):
        pred_mask = pred_cc == pred_id
        pred_voxels = int(pred_mask.sum())
        pred_volume_ml = pred_voxels * voxel_volume_ml
        overlap_gt_voxels = int(np.logical_and(pred_mask, target > 0).sum())
        is_fp_lesion = overlap_gt_voxels == 0
        if is_fp_lesion:
            false_positive_lesions += 1
        fp_lesion_rows.append({
            "case_name": case_name,
            "pred_lesion_id": pred_id,
            "pred_lesion_voxels": pred_voxels,
            "pred_lesion_volume_ml": pred_volume_ml,
            "overlap_gt_voxels": overlap_gt_voxels,
            "is_false_positive_lesion": int(is_fp_lesion),
        })

    summary = {
        "num_gt_lesions": num_gt,
        "num_detected_gt_lesions": detected_lesions,
        "num_missed_gt_lesions": missed_lesions,
        "lesion_detection_recall": detected_lesions / max(num_gt, 1),
        "num_pred_lesions": num_pred,
        "num_false_positive_lesions": false_positive_lesions,
    }
    return lesion_rows, fp_lesion_rows, summary


def summarize_lesion_histogram(lesion_rows, bins):
    hist_rows = []
    if len(lesion_rows) == 0:
        for i in range(len(bins) - 1):
            hist_rows.append({
                "volume_bin_ml": bin_label(0.0, bins, i),
                "num_gt_lesions": 0,
                "num_detected": 0,
                "num_missed": 0,
                "detection_recall": "",
            })
        return hist_rows

    volumes = np.array([float(r["gt_lesion_volume_ml"]) for r in lesion_rows], dtype=np.float64)
    detected = np.array([int(r["detected"]) for r in lesion_rows], dtype=np.int32)

    for i in range(len(bins) - 1):
        left = bins[i]
        right = bins[i + 1]
        in_bin = (volumes >= left) & (volumes < right)
        total = int(in_bin.sum())
        det = int(detected[in_bin].sum()) if total > 0 else 0
        missed = total - det
        recall = det / total if total > 0 else ""
        hist_rows.append({
            "volume_bin_ml": bin_label(0.0, bins, i),
            "num_gt_lesions": total,
            "num_detected": det,
            "num_missed": missed,
            "detection_recall": recall,
        })
    return hist_rows


def save_lesion_histogram_png(hist_rows, out_path):
    labels = [r["volume_bin_ml"] for r in hist_rows]
    detected = np.array([int(r["num_detected"]) for r in hist_rows], dtype=np.int32)
    missed = np.array([int(r["num_missed"]) for r in hist_rows], dtype=np.int32)
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 5))
    ax.bar(x, detected, label="Detected")
    ax.bar(x, missed, bottom=detected, label="Missed")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_xlabel("GT lesion volume bin (mL)")
    ax.set_ylabel("Number of GT lesions")
    ax.set_title("Lesion detection by lesion volume")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def choose_crosshair_indices(prob, pred, target=None):
    """Choose one 3D point and use its z/y/x indices for three orthogonal planes."""
    if target is not None and target.sum() > 0:
        coords = np.argwhere(target > 0)
        center = np.round(coords.mean(axis=0)).astype(int)
        z, y, x = center.tolist()
        return int(np.clip(z, 0, target.shape[0] - 1)), int(np.clip(y, 0, target.shape[1] - 1)), int(np.clip(x, 0, target.shape[2] - 1))

    if pred.sum() > 0:
        coords = np.argwhere(pred > 0)
        center = np.round(coords.mean(axis=0)).astype(int)
        z, y, x = center.tolist()
        return int(np.clip(z, 0, pred.shape[0] - 1)), int(np.clip(y, 0, pred.shape[1] - 1)), int(np.clip(x, 0, pred.shape[2] - 1))

    flat_idx = int(np.argmax(prob))
    return np.unravel_index(flat_idx, prob.shape)


def take_plane(volume, plane, z, y, x):
    if plane == "axial":
        return volume[z, :, :]
    if plane == "coronal":
        return volume[:, y, :]
    if plane == "sagittal":
        return volume[:, :, x]
    raise ValueError(f"Unknown plane: {plane}")


def display_plane(arr, plane):
    """Adjust display orientation only for visualization."""
    if plane == "axial":
        return np.flipud(arr)
    if plane == "coronal":
        return np.flipud(arr)
    if plane == "sagittal":
        return np.rot90(arr, k=2)
    raise ValueError(f"Unknown plane: {plane}")


def robust_vmin_vmax(arr):
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin, vmax = np.percentile(finite, [1, 99])
    if abs(vmax - vmin) < 1e-8:
        vmin, vmax = float(finite.min()), float(finite.max() + 1e-6)
    return float(vmin), float(vmax)


def normalize_for_display(arr):
    """Normalize one 2D slice to [0, 1] for visualization."""
    vmin, vmax = robust_vmin_vmax(arr)
    arr = np.asarray(arr, dtype=np.float32)
    arr = (arr - vmin) / max(vmax - vmin, 1e-8)
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def fuse_ct_pet_slice(ct_slice, pet_slice, ct_weight=0.5, pet_weight=0.5):
    """Fuse CT and PET slices for display using weighted average."""
    ct_norm = normalize_for_display(ct_slice)
    pet_norm = normalize_for_display(pet_slice)

    fused = ct_weight * ct_norm + pet_weight * pet_norm
    fused = np.clip(fused, 0.0, 1.0)
    return fused


def pad_to_square(arr, fill_value=0.0):
    """Pad a 2D array to a square canvas without cropping or distorting the content."""
    arr = np.asarray(arr)
    h, w = arr.shape[:2]
    side = max(h, w)
    pad_h0 = (side - h) // 2
    pad_h1 = side - h - pad_h0
    pad_w0 = (side - w) // 2
    pad_w1 = side - w - pad_w0

    if arr.ndim == 2:
        return np.pad(arr, ((pad_h0, pad_h1), (pad_w0, pad_w1)), mode="constant", constant_values=fill_value)
    if arr.ndim == 3:
        out = np.zeros((side, side, arr.shape[2]), dtype=arr.dtype)
        out[...] = fill_value
        out[pad_h0:pad_h0 + h, pad_w0:pad_w0 + w, :] = arr
        return out
    raise ValueError(f"Expected 2D or 3D array, got shape {arr.shape}")


def make_display_slice(slice2d, plane, fill_value=0.0):
    return pad_to_square(display_plane(slice2d, plane), fill_value=fill_value)


def overlay_mask(ax, base, mask, alpha, title, plane, base_cmap="gray"):
    vmin, vmax = robust_vmin_vmax(base)
    base_show = make_display_slice(base, plane, fill_value=vmin)
    mask_show = make_display_slice(mask, plane, fill_value=0)

    ax.imshow(base_show, cmap=base_cmap, vmin=vmin, vmax=vmax, aspect="equal")

    overlay = np.zeros((mask_show.shape[0], mask_show.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = 0.0
    overlay[..., 1] = 1.0
    overlay[..., 2] = 0.0
    overlay[..., 3] = (mask_show > 0).astype(np.float32) * alpha
    ax.imshow(overlay, interpolation="none", aspect="equal")

    ax.set_title(title)
    ax.axis("off")


def overlay_mask_on_fused(ax, ct_slice, pet_slice, mask, alpha, title, plane,
                          ct_weight=0.5, pet_weight=0.5):
    """
    Show GT/pred mask on fused CT+PET image.
    The fused image and mask are both padded to a square canvas.
    """
    fused = fuse_ct_pet_slice(
        ct_slice,
        pet_slice,
        ct_weight=ct_weight,
        pet_weight=pet_weight
    )

    # Apply the same orientation correction and square padding as other columns.
    fused_show = make_display_slice(fused, plane, fill_value=0.0)
    mask_show = make_display_slice(mask, plane, fill_value=0)

    ax.imshow(
        fused_show,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        aspect="equal"
    )

    overlay = np.zeros(
        (mask_show.shape[0], mask_show.shape[1], 4),
        dtype=np.float32
    )
    overlay[..., 0] = 0.0
    overlay[..., 1] = 1.0
    overlay[..., 2] = 0.0
    overlay[..., 3] = (mask_show > 0).astype(np.float32) * alpha

    ax.imshow(
        overlay,
        interpolation="none",
        aspect="equal"
    )

    ax.set_title(title)
    ax.axis("off")


def overlay_mask_on_fused1(ax, ct_slice, pet_slice, mask, alpha, title, plane,
                          ct_weight=0.5, pet_weight=0.5):
    """
    Show GT/pred mask on fused CT+PET image.
    """
    fused = fuse_ct_pet_slice(ct_slice, pet_slice, ct_weight=ct_weight, pet_weight=pet_weight)
    fused_disp = display_plane(fused, plane)
    mask_disp = display_plane(mask, plane)

    ax.imshow(fused_disp, cmap="gray", vmin=0.0, vmax=1.0)

    overlay = np.zeros((mask_disp.shape[0], mask_disp.shape[1], 4), dtype=np.float32)
    overlay[..., 0] = 0.0
    overlay[..., 1] = 1.0
    overlay[..., 2] = 0.0
    overlay[..., 3] = (mask_disp > 0).astype(np.float32) * alpha

    ax.imshow(overlay, interpolation="none")
    ax.set_title(title)
    ax.axis("off")


def show_mask(ax, mask, title, plane):
    mask_show = make_display_slice(mask, plane, fill_value=0)
    ax.imshow(mask_show, cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax.set_title(title)
    ax.axis("off")


def save_three_plane_png(args, out_path, image_orig, prob_orig, pred_orig, target_orig=None):
    """Save axial/coronal/sagittal visualization in original-volume space."""
    ct = image_orig[args.ct_channel]
    pet_idx = args.pet_channel if image_orig.shape[0] > args.pet_channel else min(1, image_orig.shape[0] - 1)
    pet = image_orig[pet_idx]

    z, y, x = choose_crosshair_indices(prob=prob_orig, pred=pred_orig, target=target_orig)
    planes = [
        ("axial", f"Axial z={z}"),
        ("coronal", f"Coronal y={y}"),
        ("sagittal", f"Sagittal x={x}"),
    ]

    has_gt = target_orig is not None
    ncols = 5 if has_gt else 3
    fig, axes = plt.subplots(3, ncols, figsize=(4 * ncols, 12))
    if ncols == 1:
        axes = axes[:, None]

    for row, (plane, plane_title) in enumerate(planes):
        ct_slice = take_plane(ct, plane, z, y, x)
        pet_slice = take_plane(pet, plane, z, y, x)
        pred_slice = take_plane(pred_orig, plane, z, y, x)
        gt_slice = take_plane(target_orig, plane, z, y, x) if has_gt else None

        col = 0
        overlay_mask(axes[row, col], ct_slice, pred_slice, args.overlay_alpha, f"{plane_title}\nCT + Pred", plane=plane)
        col += 1

        overlay_mask(axes[row, col], pet_slice, pred_slice, args.overlay_alpha, "PET + Pred", plane=plane)
        col += 1

        if has_gt:
            overlay_mask_on_fused(axes[row, col], ct_slice, pet_slice, gt_slice, args.overlay_alpha, "CT/PET Fusion + GT", plane=plane, ct_weight=0.5, pet_weight=0.5)
            col += 1

        show_mask(axes[row, col], pred_slice, "Pred Mask", plane=plane)
        col += 1

        if has_gt:
            show_mask(axes[row, col], gt_slice, "GT Mask", plane=plane)

    fig.suptitle(
        f"Segmentation overlay in original space | threshold={args.threshold}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_nifti(path, volume, affine, dtype):
    if nib is None:
        raise ImportError("nibabel is required for NIfTI export. Install it with: pip install nibabel")
    nii = nib.Nifti1Image(volume.astype(dtype), affine)
    nib.save(nii, path)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_metric_summary(metric_rows):
    valid = [r for r in metric_rows if r["dice"] != ""]
    if len(valid) == 0:
        return []

    def arr(key):
        return np.asarray([float(r[key]) for r in valid], dtype=np.float64)

    keys = [
        "dice", "fpv_ml", "fnv_ml", "pred_volume_ml", "gt_volume_ml",
        "num_gt_lesions", "num_detected_gt_lesions", "num_missed_gt_lesions",
        "lesion_detection_recall", "num_pred_lesions", "num_false_positive_lesions",
    ]
    rows = []
    for key in keys:
        values = arr(key)
        rows.append({
            "metric": key,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        })
    return rows


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    args = copy_args_from_checkpoint(args, ckpt)

    if args.test_batch_size != 1:
        raise ValueError("Full-volume export requires --test_batch_size 1.")

    os.makedirs(args.out_dir, exist_ok=True)
    png_dir = os.path.join(args.out_dir, "png_three_plane_original_space")
    npz_dir = os.path.join(args.out_dir, "npz_original_space")
    target_npz_dir = os.path.join(args.out_dir, "npz_target_224_space")
    nii_dir = os.path.join(args.out_dir, "nii_original_space")
    lesion_dir = os.path.join(args.out_dir, "lesion_analysis")

    if args.save_png:
        os.makedirs(png_dir, exist_ok=True)
    if args.save_npz:
        os.makedirs(npz_dir, exist_ok=True)
    if args.save_target_npz:
        os.makedirs(target_npz_dir, exist_ok=True)
    if args.save_nii or args.save_prob_nii:
        os.makedirs(nii_dir, exist_ok=True)
    os.makedirs(lesion_dir, exist_ok=True)

    model = build_model(args, ckpt, device)

    label_dir = args.test_label_path if args.test_label_path else None
    dataset = ExportFullImagePatchDataset3D(
        image_dir=args.test_data_path,
        label_dir=label_dir,
        patch_size=args.patch_size,
        stride=args.stride,
        target_size=args.target_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        pin_memory=True,
    )

    metric_rows = []
    all_lesion_rows = []
    all_fp_lesion_rows = []
    lesion_bins = parse_bins_ml(args.lesion_bins_ml)

    print(f"Device: {device}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Output directory: {args.out_dir}")
    print(f"patch_size={args.patch_size}, stride={args.stride}, base_channels={args.base_channels}, latent_dim={args.latent_dim}")

    for batch in tqdm(loader, desc="Exporting segmentation"):
        x_target, logit_target, prob_target, pred_target = infer_one_volume(args, model, batch, device)

        file_name = batch.get("file_name", ["case"])[0]
        case_name = batch.get("case_name", [get_case_name(file_name)])[0]
        image_path = batch.get("image_path", [os.path.join(args.test_data_path, file_name)])[0]
        original_shape = tuple(batch["original_shape"][0].cpu().numpy().tolist())

        image_target_np = x_target.detach().cpu().numpy().astype(np.float32)
        prob_target_np = prob_target[0].detach().cpu().numpy().astype(np.float32)
        pred_target_np = pred_target[0].detach().cpu().numpy().astype(np.uint8)

        image_orig_np = batch["image_orig"][0].cpu().numpy().astype(np.float32)
        if tuple(prob_target_np.shape) != tuple(original_shape):
            prob_orig_np = resize_volume_np(prob_target_np, original_shape, mode="trilinear").astype(np.float32)
            pred_orig_np = (prob_orig_np > args.threshold).astype(np.uint8)
        else:
            prob_orig_np = prob_target_np.astype(np.float32)
            pred_orig_np = pred_target_np.astype(np.uint8)

        y_orig = get_label_orig_from_batch(batch)
        target_orig_np = None
        if y_orig is not None:
            target_orig_np = (y_orig[0].cpu().numpy() > 0.5).astype(np.uint8)
            if tuple(target_orig_np.shape) != tuple(original_shape):
                target_orig_np = resize_volume_np(target_orig_np, original_shape, mode="nearest").astype(np.uint8)

        # affine = load_affine_from_npz(image_path)
        # voxel_volume_ml = voxel_volume_ml_from_affine(affine)
        #affine, spacing, spacing_source = load_geometry_from_npz(
        #    image_path,
        #    strict=False
        #)
        affine, spacing_xyz, spacing_source = load_geometry_from_npz(
            image_path
        )

         #voxel_volume_ml = voxel_volume_ml_from_affine(affine)
        voxel_volume_ml = float(np.prod(spacing_xyz) / 1000.0)

        print(
            f"[SpacingCheck] {case_name} | "
            f"source={spacing_source} | "
            f"spacing_xyz={spacing_xyz.tolist()} | "
            f"spacing_zyx={spacing_xyz[::-1].tolist()} | "
            f"voxel_volume_ml={voxel_volume_ml:.8f}"
        )
        spacing = spacing_xyz

        check_spacing_info(
            case_name=case_name,
            spacing=spacing,
            affine=affine,
            voxel_volume_ml=voxel_volume_ml,
            original_shape=original_shape,
        )


        if args.save_png:
            png_path = os.path.join(png_dir, f"{case_name}_three_plane_overlay_original_space.png")
            save_three_plane_png(
                args=args,
                out_path=png_path,
                image_orig=image_orig_np,
                prob_orig=prob_orig_np,
                pred_orig=pred_orig_np,
                target_orig=target_orig_np,
            )

        if args.save_npz:
            npz_path = os.path.join(npz_dir, f"{case_name}_segmentation_result_original_space.npz")
            save_items = {
                "image": image_orig_np,
                "prob": prob_orig_np.astype(np.float32),
                "pred": pred_orig_np.astype(np.uint8),
                "threshold": np.array(args.threshold, dtype=np.float32),
                "affine": affine.astype(np.float32),
                "voxel_volume_ml": np.array(voxel_volume_ml, dtype=np.float32),
                "spacing_0_mm": float(spacing[0]),
                "spacing_1_mm": float(spacing[1]),
                "spacing_2_mm": float(spacing[2]),
                "spacing_source": spacing_source,
                "case_name": case_name,
                "file_name": file_name,
            }
            if target_orig_np is not None:
                save_items["label"] = target_orig_np.astype(np.uint8)
            np.savez_compressed(npz_path, **save_items)

        if args.save_target_npz:
            npz_path = os.path.join(target_npz_dir, f"{case_name}_segmentation_result_target_224_space.npz")
            save_items = {
                "image": image_target_np,
                "prob": prob_target_np.astype(np.float32),
                "pred": pred_target_np.astype(np.uint8),
                "threshold": np.array(args.threshold, dtype=np.float32),
                "case_name": case_name,
                "file_name": file_name,
            }
            np.savez_compressed(npz_path, **save_items)

        if args.save_nii:
            save_nifti(os.path.join(nii_dir, f"{case_name}_pred_mask_original_space.nii.gz"), pred_orig_np, affine, np.uint8)
            if target_orig_np is not None:
                save_nifti(os.path.join(nii_dir, f"{case_name}_gt_label_original_space.nii.gz"), target_orig_np, affine, np.uint8)

        if args.save_prob_nii:
            save_nifti(os.path.join(nii_dir, f"{case_name}_pred_prob_original_space.nii.gz"), prob_orig_np, affine, np.float32)

        row = {
            "case_name": case_name,
            "file_name": file_name,
            "dice": "",
            "fp_voxels": "",
            "fn_voxels": "",
            "fpv_ml": "",
            "fnv_ml": "",
            "pred_volume_ml": int(pred_orig_np.sum()) * voxel_volume_ml,
            "gt_volume_ml": "",
            "voxel_volume_ml": voxel_volume_ml,
            "spacing_0_mm": float(spacing[0]),
            "spacing_1_mm": float(spacing[1]),
            "spacing_2_mm": float(spacing[2]),
            "spacing_source": spacing_source,
            "threshold": args.threshold,
            "num_gt_lesions": "",
            "num_detected_gt_lesions": "",
            "num_missed_gt_lesions": "",
            "lesion_detection_recall": "",
            "num_pred_lesions": "",
            "num_false_positive_lesions": "",
        }

        if target_orig_np is not None:
            dice_value = dice_np(pred_orig_np, target_orig_np)
            fp, fn, fpv_ml, fnv_ml = voxel_error_stats_ml(pred_orig_np, target_orig_np, voxel_volume_ml)
            row.update({
                "dice": dice_value,
                "fp_voxels": fp,
                "fn_voxels": fn,
                "fpv_ml": fpv_ml,
                "fnv_ml": fnv_ml,
                "gt_volume_ml": int(target_orig_np.sum()) * voxel_volume_ml,
            })

            if ndi is not None:
                lesion_rows, fp_lesion_rows, lesion_summary = compute_lesion_detection(
                    case_name=case_name,
                    pred=pred_orig_np,
                    target=target_orig_np,
                    voxel_volume_ml=voxel_volume_ml,
                    args=args,
                )
                all_lesion_rows.extend(lesion_rows)
                all_fp_lesion_rows.extend(fp_lesion_rows)
                row.update(lesion_summary)
            else:
                print("Warning: scipy is not installed. Lesion-level statistics were skipped.")

        metric_rows.append(row)

    metric_fieldnames = [
        "case_name", "file_name", "dice", "fp_voxels", "fn_voxels", "fpv_ml", "fnv_ml",
        "pred_volume_ml", "gt_volume_ml", "voxel_volume_ml", "spacing_0_mm", "spacing_1_mm", "spacing_2_mm", "spacing_source", "threshold",
        "num_gt_lesions", "num_detected_gt_lesions", "num_missed_gt_lesions", "lesion_detection_recall",
        "num_pred_lesions", "num_false_positive_lesions",
    ]
    metrics_csv_path = os.path.join(args.out_dir, "segmentation_metrics_original_space.csv")
    write_csv(metrics_csv_path, metric_rows, metric_fieldnames)

    summary_rows = make_metric_summary(metric_rows)
    if len(summary_rows) > 0:
        summary_csv_path = os.path.join(args.out_dir, "segmentation_metrics_summary_original_space.csv")
        write_csv(summary_csv_path, summary_rows, ["metric", "mean", "std", "median", "min", "max"])

    if len(all_lesion_rows) > 0:
        lesion_csv_path = os.path.join(lesion_dir, "lesion_detection_per_gt_lesion.csv")
        write_csv(
            lesion_csv_path,
            all_lesion_rows,
            ["case_name", "gt_lesion_id", "gt_lesion_voxels", "gt_lesion_volume_ml", "overlap_voxels", "overlap_ratio", "detected"],
        )

        fp_lesion_csv_path = os.path.join(lesion_dir, "predicted_lesion_false_positive_analysis.csv")
        write_csv(
            fp_lesion_csv_path,
            all_fp_lesion_rows,
            ["case_name", "pred_lesion_id", "pred_lesion_voxels", "pred_lesion_volume_ml", "overlap_gt_voxels", "is_false_positive_lesion"],
        )

        hist_rows = summarize_lesion_histogram(all_lesion_rows, lesion_bins)
        hist_csv_path = os.path.join(lesion_dir, "lesion_detection_histogram_by_volume.csv")
        write_csv(hist_csv_path, hist_rows, ["volume_bin_ml", "num_gt_lesions", "num_detected", "num_missed", "detection_recall"])

        hist_png_path = os.path.join(lesion_dir, "lesion_detection_histogram_by_volume.png")
        save_lesion_histogram_png(hist_rows, hist_png_path)

    print(f"Saved metrics CSV: {metrics_csv_path}")
    print("Done.")


if __name__ == "__main__":
    main()
