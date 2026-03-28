import argparse
from pathlib import Path
import pickle
import blosc2
import numpy as np
import SimpleITK as sitk


DATA_SUFFIX = ".b2nd"
SEG_SUFFIX = "_seg.b2nd"
PKL_SUFFIX = ".pkl"

CT_CHANNEL_INDEX = 0
PET_CHANNEL_INDEX = 1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch export nnU-Net v2 preprocessed .b2nd/.pkl cases to CT/PET/SEG .nii.gz"
    )
    parser.add_argument(
        "--source_root",
        type=str,
        required=True,
        help="Root folder of nnUNet_preprocessed or one of its subfolders",
    )
    parser.add_argument(
        "--target_root",
        type=str,
        required=True,
        help="Output folder for exported CT/PET/SEG files",
    )
    parser.add_argument(
        "--properties_mode",
        type=str,
        default="inline",
        choices=["inline", "separate"],
        help="inline: do not save properties separately; separate: save properties as an extra _properties.pkl",
    )
    return parser.parse_args()


def get_case_id_from_data_file(data_file: Path) -> str:
    return data_file.name[:-len(DATA_SUFFIX)]


def collect_cases(source_root: Path):
    cases = {}

    for data_file in source_root.rglob(f"*{DATA_SUFFIX}"):
        if data_file.name.endswith(SEG_SUFFIX):
            continue

        case_id = get_case_id_from_data_file(data_file)
        seg_file = data_file.with_name(case_id + SEG_SUFFIX)
        pkl_file = data_file.with_name(case_id + PKL_SUFFIX)

        cases[case_id] = {
            "data": data_file,
            "seg": seg_file if seg_file.exists() else None,
            "pkl": pkl_file if pkl_file.exists() else None,
        }

    return cases


def load_b2nd_array(file_path: Path) -> np.ndarray:
    arr = blosc2.open(str(file_path))
    return arr[:]


def load_properties(pkl_path: Path):
    if pkl_path is None or not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        props = pickle.load(f)
    return props


def get_spacing_xyz(props):
    """
    nnU-Net properties usually store spacing in z, y, x order.
    SimpleITK expects x, y, z order.
    """
    if props is None:
        return None

    for key in ["spacing", "spacing_after_resampling", "current_spacing"]:
        if key in props:
            spacing_zyx = props[key]
            if spacing_zyx is not None and len(spacing_zyx) == 3:
                return tuple(float(x) for x in spacing_zyx[::-1])

    return None


def save_nifti(volume_zyx: np.ndarray, save_path: Path, spacing_xyz=None, is_label=False):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if is_label:
        volume_zyx = volume_zyx.astype(np.int16)
    else:
        volume_zyx = volume_zyx.astype(np.float32)

    image = sitk.GetImageFromArray(volume_zyx)

    if spacing_xyz is not None:
        image.SetSpacing(spacing_xyz)

    sitk.WriteImage(image, str(save_path))


def export_cases(source_root: Path, target_root: Path, properties_mode: str):
    cases = collect_cases(source_root)
    print(f"Found {len(cases)} cases")

    if len(cases) == 0:
        print("No valid cases found.")
        return

    ct_dir = target_root / "CT"
    pet_dir = target_root / "PET"
    seg_dir = target_root / "SEG"
    props_dir = target_root / "properties"

    for i, (case_id, files) in enumerate(sorted(cases.items()), start=1):
        print(f"\n[{i}/{len(cases)}] Processing {case_id}")

        try:
            data = load_b2nd_array(files["data"])
            props = load_properties(files["pkl"])
            spacing_xyz = get_spacing_xyz(props)

            print(f"  data shape: {data.shape}, dtype: {data.dtype}")

            if data.ndim != 4:
                print("  Skip: data is not 4D [C, Z, Y, X]")
                continue

            if data.shape[0] <= max(CT_CHANNEL_INDEX, PET_CHANNEL_INDEX):
                print("  Skip: data does not contain enough channels")
                continue

            ct = data[CT_CHANNEL_INDEX]
            pet = data[PET_CHANNEL_INDEX]

            ct_path = ct_dir / f"{case_id}_CT.nii.gz"
            pet_path = pet_dir / f"{case_id}_PET.nii.gz"

            save_nifti(ct, ct_path, spacing_xyz=spacing_xyz, is_label=False)
            save_nifti(pet, pet_path, spacing_xyz=spacing_xyz, is_label=False)

            print(f"  saved CT : {ct_path}")
            print(f"  saved PET: {pet_path}")

            if files["seg"] is not None:
                seg = load_b2nd_array(files["seg"])
                print(f"  seg shape : {seg.shape}, dtype: {seg.dtype}")

                if seg.ndim == 4 and seg.shape[0] == 1:
                    seg = seg[0]
                elif seg.ndim == 3:
                    pass
                else:
                    print("  Skip SEG: unexpected seg shape")
                    seg = None

                if seg is not None:
                    seg_path = seg_dir / f"{case_id}_SEG.nii.gz"
                    save_nifti(seg, seg_path, spacing_xyz=spacing_xyz, is_label=True)
                    print(f"  saved SEG: {seg_path}")
            else:
                print("  No SEG file found for this case")

            if properties_mode == "separate" and props is not None:
                props_dir.mkdir(parents=True, exist_ok=True)
                props_path = props_dir / f"{case_id}_properties.pkl"
                with open(props_path, "wb") as f:
                    pickle.dump(props, f)
                print(f"  saved properties: {props_path}")

        except Exception as e:
            print(f"  Error processing {case_id}: {e}")


def main():
    args = parse_args()
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)

    export_cases(
        source_root=source_root,
        target_root=target_root,
        properties_mode=args.properties_mode,
    )


if __name__ == "__main__":
    main()