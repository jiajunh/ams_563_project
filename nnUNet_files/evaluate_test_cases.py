import os
import csv
import math
import cc3d
import nibabel as nib
import numpy as np


def nii2numpy(nii_path: str):
    """Load a nifti file and extract the voxel volume in mL."""
    mask_nii = nib.load(nii_path)
    mask = (mask_nii.get_fdata() > 0).astype(np.uint8)
    pixdim = mask_nii.header["pixdim"]
    voxel_vol = float(pixdim[1] * pixdim[2] * pixdim[3] / 1000.0)  # mm^3 -> mL
    return mask, voxel_vol


def count_false_positives(*, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Count volume in voxels of predicted connected components that do not overlap any GT lesion."""
    if prediction.sum() == 0:
        return 0.0

    connected_components = cc3d.connected_components(
        prediction.astype(np.uint8), connectivity=18
    )
    false_positives = 0

    for idx in range(1, connected_components.max() + 1):
        component_mask = (connected_components == idx)
        if (component_mask & (ground_truth > 0)).sum() == 0:
            false_positives += int(component_mask.sum())

    return float(false_positives)


def count_false_negatives(*, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Count volume in voxels of GT connected components completely missed by prediction."""
    if ground_truth.sum() == 0:
        return np.nan

    gt_components = cc3d.connected_components(
        ground_truth.astype(np.uint8), connectivity=18
    )
    false_negatives = 0

    for idx in range(1, gt_components.max() + 1):
        component_mask = (gt_components == idx)
        if (component_mask & (prediction > 0)).sum() == 0:
            false_negatives += int(component_mask.sum())

    return float(false_negatives)


def calc_dice_score(*, prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """Calculate voxel-wise Dice score. Return NaN if GT is empty."""
    if ground_truth.sum() == 0:
        return np.nan

    intersection = ((ground_truth > 0) & (prediction > 0)).sum()
    union = (ground_truth > 0).sum() + (prediction > 0).sum()

    if union == 0:
        return np.nan

    dice_score = 2.0 * intersection / union
    return float(dice_score)


def compute_metrics(nii_gt_path: str, nii_pred_path: str):
    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, _ = nii2numpy(nii_pred_path)

    false_neg_vol = count_false_negatives(
        prediction=pred_array, ground_truth=gt_array
    ) * voxel_vol

    false_pos_vol = count_false_positives(
        prediction=pred_array, ground_truth=gt_array
    ) * voxel_vol

    dice_sc = calc_dice_score(prediction=pred_array, ground_truth=gt_array)

    return dice_sc, false_pos_vol, false_neg_vol


def safe_mean(values):
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if len(vals) == 0:
        return float("nan")
    return float(np.mean(vals))


def evaluate_folder(gt_dir: str, pred_dir: str, output_csv: str):
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith(".nii.gz")])

    rows = []
    dice_list = []
    fpv_list = []
    fnv_list = []

    missing_gt = []

    for fname in pred_files:
        pred_path = os.path.join(pred_dir, fname)
        gt_path = os.path.join(gt_dir, fname)

        if not os.path.exists(gt_path):
            missing_gt.append(fname)
            continue

        dice_sc, fpv, fnv = compute_metrics(gt_path, pred_path)

        rows.append({
            "Case": fname,
            "Dice": dice_sc,
            "FPV_mL": fpv,
            "FNV_mL": fnv,
            "PredictionFile": pred_path,
            "GroundTruthFile": gt_path,
        })

        dice_list.append(dice_sc)
        fpv_list.append(fpv)
        fnv_list.append(fnv)

        print(f"{fname}: Dice={dice_sc:.4f} | FPV={fpv:.4f} mL | FNV={fnv:.4f} mL")

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Case", "Dice", "FPV_mL", "FNV_mL", "PredictionFile", "GroundTruthFile"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved per-case results to:", output_csv)
    print(f"Evaluated {len(rows)} cases")

    if missing_gt:
        print("\nMissing GT for these prediction files:")
        for x in missing_gt:
            print("  ", x)

    print("\nSummary:")
    print(f"Mean Dice   = {safe_mean(dice_list):.4f}")
    print(f"Mean FPV mL = {safe_mean(fpv_list):.4f}")
    print(f"Mean FNV mL = {safe_mean(fnv_list):.4f}")


if __name__ == "__main__":
    gt_dir = "/PETCT/dataset/fdg-pet-ct-lesions/nnUNet_raw/Dataset001_FDGPETCT_autopet3/labelsTs"
    pred_dir = "/PETCT/models/autoPETIII-main/nnUNet_results/nnUNet_3d_fullres_ResEnvUNet"
    output_csv = "/PETCT/models/autoPETIII-main/nnUNet_results/nnUNet_3d_fullres_ResEnvUNet/per_case_metrics.csv"

    evaluate_folder(gt_dir, pred_dir, output_csv)