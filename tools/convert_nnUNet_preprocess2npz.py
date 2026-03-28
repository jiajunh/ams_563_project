import os
import pickle
import argparse
from pathlib import Path

import numpy as np
import blosc2


def find_cases_recursive(preprocessed_root: Path):
    """
    Recursively find nnU-Net v2 preprocessed cases.
    A valid case is identified by:
      - case_id.b2nd
      - case_id_seg.b2nd
      - case_id.pkl
    """
    cases = []

    for data_file in preprocessed_root.rglob("*.b2nd"):
        if data_file.name.endswith("_seg.b2nd"):
            continue

        case_id = data_file.stem
        seg_file = data_file.with_name(f"{case_id}_seg.b2nd")
        pkl_file = data_file.with_suffix(".pkl")

        if not seg_file.exists():
            print(f"[Skip] Missing seg file: {seg_file}")
            continue

        if not pkl_file.exists():
            print(f"[Skip] Missing pkl file: {pkl_file}")
            continue

        cases.append((data_file, seg_file, pkl_file, case_id))

    return sorted(cases, key=lambda x: str(x[0]))


def load_case(data_file: Path, seg_file: Path, pkl_file: Path):
    """
    Load one nnU-Net v2 case from .b2nd + .pkl.
    """
    data = blosc2.open(urlpath=str(data_file), mode="r")[:]
    seg = blosc2.open(urlpath=str(seg_file), mode="r")[:]

    with open(pkl_file, "rb") as f:
        properties = pickle.load(f)

    return data, seg, properties


def save_case_to_npz(
    out_npz: Path,
    data: np.ndarray,
    seg: np.ndarray,
    properties: dict,
    store_properties_in_npz: bool = True,
):
    """
    Save one case to .npz.
    If store_properties_in_npz is True, properties are stored as an object array.
    """
    out_npz.parent.mkdir(parents=True, exist_ok=True)

    if store_properties_in_npz:
        np.savez_compressed(
            out_npz,
            data=data.astype(np.float32),
            seg=seg.astype(np.int16),
            properties=np.array(properties, dtype=object),
        )
    else:
        np.savez_compressed(
            out_npz,
            data=data.astype(np.float32),
            seg=seg.astype(np.int16),
        )

        out_pkl = out_npz.with_name(out_npz.stem + "_properties.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(properties, f)


def export_all_cases(
    source_root: str,
    target_root: str,
    store_properties_in_npz: bool = True,
):
    """
    Recursively export all nnU-Net v2 preprocessed cases from source_root
    into target_root, preserving relative folder structure.
    """
    source_root = Path(source_root).resolve()
    target_root = Path(target_root).resolve()

    cases = find_cases_recursive(source_root)
    print(f"Found {len(cases)} valid cases.")

    num_ok = 0
    num_fail = 0

    for data_file, seg_file, pkl_file, case_id in cases:
        try:
            relative_parent = data_file.parent.relative_to(source_root)
            out_dir = target_root / relative_parent
            out_npz = out_dir / f"{case_id}.npz"

            data, seg, properties = load_case(data_file, seg_file, pkl_file)

            save_case_to_npz(
                out_npz=out_npz,
                data=data,
                seg=seg,
                properties=properties,
                store_properties_in_npz=store_properties_in_npz,
            )

            print(
                f"[Saved] {case_id} | "
                f"data={data.shape} {data.dtype} | "
                f"seg={seg.shape} {seg.dtype} -> {out_npz}"
            )
            num_ok += 1

        except Exception as e:
            print(f"[Failed] {case_id}: {e}")
            num_fail += 1

    print("\nDone.")
    print(f"Success: {num_ok}")
    print(f"Failed : {num_fail}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch export nnU-Net v2 preprocessed .b2nd/.pkl cases to .npz"
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
        help="Output folder for exported .npz files",
    )
    parser.add_argument(
        "--properties_mode",
        type=str,
        default="inline",
        choices=["inline", "separate"],
        help="inline: store properties inside npz; separate: save properties as an extra _properties.pkl",
    )

    args = parser.parse_args()

    store_properties_in_npz = args.properties_mode == "inline"

    export_all_cases(
        source_root=args.source_root,
        target_root=args.target_root,
        store_properties_in_npz=store_properties_in_npz,
    )


if __name__ == "__main__":
    main()