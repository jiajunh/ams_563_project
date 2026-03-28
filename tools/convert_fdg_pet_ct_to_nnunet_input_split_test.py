from pathlib import Path
import shutil
import csv
import json
import random


def _extract_study_short_id(study_folder_name: str) -> str:
    """
    Extract the short study ID from the study folder name.

    Example:
        '12-29-2002-NA-Unspecified CT ABDOMEN-93772' -> '93772'
        '03-31-2003-NA-PET-CT Ganzkoerper  primaer mit KM-22165' -> '22165'
    """
    parts = study_folder_name.rsplit("-", 1)
    if len(parts) == 2 and parts[1]:
        return parts[1]
    return study_folder_name.replace(" ", "_")


def _safe_copy_or_symlink(src: Path, dst: Path, use_symlink: bool):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if use_symlink:
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def convert_fdg_pet_ct_to_nnunet_input(
    src_root,
    out_root,
    dataset_id=1,
    dataset_name_suffix="FDGPETCT_autopet3",
    use_symlink=False,
    test_ratio=0.2,
    random_seed=42,
    split_by_subject=True,
    generate_dataset_json=True,
    verbose=True,
):
    """
    Convert FDG-PET-CT-Lesions to nnU-Net input format and split into train/test.

    Output:
        nnUNet_raw/
        └── Dataset001_FDGPETCT_autopet3/
            ├── imagesTr/
            ├── labelsTr/
            ├── imagesTs/
            ├── labelsTs/
            ├── dataset.json
            ├── case_mapping.csv
            └── split_summary.csv

    Naming:
        {SubjectID}_{StudyShortID}_0000.nii.gz = CTres
        {SubjectID}_{StudyShortID}_0001.nii.gz = SUV
        {SubjectID}_{StudyShortID}.nii.gz      = SEG
    """
    src_root = Path(src_root)
    dataset_name = f"Dataset{dataset_id:03d}_{dataset_name_suffix}"
    out_root = Path(out_root) / dataset_name

    imagesTr = out_root / "imagesTr"
    labelsTr = out_root / "labelsTr"
    imagesTs = out_root / "imagesTs"
    labelsTs = out_root / "labelsTs"

    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    imagesTs.mkdir(parents=True, exist_ok=True)
    labelsTs.mkdir(parents=True, exist_ok=True)

    subject_dirs = [
        d for d in sorted(src_root.iterdir())
        if d.is_dir() and d.name.startswith("PETCT_")
    ]

    if verbose:
        print(f"Found {len(subject_dirs)} PETCT subject folders.")
        print("-" * 80)

    # Step 1: collect all valid studies
    all_cases = []
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name

        for study_idx, study_dir in enumerate(sorted(subject_dir.iterdir())):
            if not study_dir.is_dir():
                continue

            ct_file = study_dir / "CTres.nii.gz"
            pet_file = study_dir / "SUV.nii.gz"
            seg_file = study_dir / "SEG.nii.gz"

            # Training set must have CT + PET + SEG
            if not (ct_file.exists() and pet_file.exists() and seg_file.exists()):
                continue

            study_short_id = _extract_study_short_id(study_dir.name)
            case_id = f"{subject_id}_{study_short_id}"

            all_cases.append({
                "case_id": case_id,
                "subject_id": subject_id,
                "study_index_within_subject": study_idx,
                "study_short_id": study_short_id,
                "study_folder": study_dir.name,
                "src_ctres": str(ct_file),
                "src_suv": str(pet_file),
                "src_seg": str(seg_file),
            })

    if len(all_cases) == 0:
        raise RuntimeError("No valid cases found with CTres.nii.gz + SUV.nii.gz + SEG.nii.gz")

    if verbose:
        print(f"Collected {len(all_cases)} valid labeled studies.")

    # Step 2: split train/test
    rng = random.Random(random_seed)

    if split_by_subject:
        unique_subjects = sorted({row["subject_id"] for row in all_cases})
        rng.shuffle(unique_subjects)

        n_test_subjects = max(1, round(len(unique_subjects) * test_ratio))
        test_subjects = set(unique_subjects[:n_test_subjects])

        for row in all_cases:
            row["split"] = "test" if row["subject_id"] in test_subjects else "train"
    else:
        indices = list(range(len(all_cases)))
        rng.shuffle(indices)
        n_test_cases = max(1, round(len(all_cases) * test_ratio))
        test_idx = set(indices[:n_test_cases])

        for i, row in enumerate(all_cases):
            row["split"] = "test" if i in test_idx else "train"

    # Step 3: export files
    num_train = 0
    num_test = 0

    for i, row in enumerate(all_cases, start=1):
        case_id = row["case_id"]
        ct_file = Path(row["src_ctres"])
        pet_file = Path(row["src_suv"])
        seg_file = Path(row["src_seg"])

        if row["split"] == "train":
            out_ct = imagesTr / f"{case_id}_0000.nii.gz"
            out_pet = imagesTr / f"{case_id}_0001.nii.gz"
            out_seg = labelsTr / f"{case_id}.nii.gz"
            num_train += 1
        else:
            out_ct = imagesTs / f"{case_id}_0000.nii.gz"
            out_pet = imagesTs / f"{case_id}_0001.nii.gz"
            out_seg = labelsTs / f"{case_id}.nii.gz"
            num_test += 1

        _safe_copy_or_symlink(ct_file, out_ct, use_symlink)
        _safe_copy_or_symlink(pet_file, out_pet, use_symlink)
        _safe_copy_or_symlink(seg_file, out_seg, use_symlink)

        row["out_ct_0000"] = str(out_ct)
        row["out_pet_0001"] = str(out_pet)
        row["out_seg"] = str(out_seg) if out_seg is not None else ""
        row["out_split"] = row["split"]

        if verbose and (i % 20 == 0 or i == len(all_cases)):
            print(f"Exported {i}/{len(all_cases)} cases...")

    # Step 4: write case mapping csv
    mapping_csv = out_root / "case_mapping.csv"
    with open(mapping_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case_id",
                "subject_id",
                "study_index_within_subject",
                "study_short_id",
                "study_folder",
                "split",
                "src_ctres",
                "src_suv",
                "src_seg",
                "out_ct_0000",
                "out_pet_0001",
                "out_seg",
                "out_split",
            ],
        )
        writer.writeheader()
        writer.writerows(all_cases)

    # Step 5: write split summary
    split_summary_csv = out_root / "split_summary.csv"
    subject_train = len({r["subject_id"] for r in all_cases if r["split"] == "train"})
    subject_test = len({r["subject_id"] for r in all_cases if r["split"] == "test"})

    with open(split_summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["num_total_cases", len(all_cases)])
        writer.writerow(["num_train_cases", num_train])
        writer.writerow(["num_test_cases", num_test])
        writer.writerow(["num_train_subjects", subject_train])
        writer.writerow(["num_test_subjects", subject_test])
        writer.writerow(["split_by_subject", split_by_subject])
        writer.writerow(["test_ratio", test_ratio])
        writer.writerow(["random_seed", random_seed])

    # Step 6: write dataset.json for nnU-Net v2
    if generate_dataset_json:
        dataset_json = {
            "channel_names": {
                "0": "CT",
                "1": "PET"
            },
            "labels": {
                "background": 0,
                "lesion": 1
            },
            "numTraining": num_train,
            "file_ending": ".nii.gz"
        }

        with open(out_root / "dataset.json", "w", encoding="utf-8") as f:
            json.dump(dataset_json, f, indent=2)

    print("-" * 80)
    print(f"Done.")
    print(f"Dataset folder : {out_root}")
    print(f"imagesTr       : {imagesTr}")
    print(f"labelsTr       : {labelsTr}")
    print(f"imagesTs       : {imagesTs}")
    print(f"labelsTs       : {labelsTs}")
    print(f"Training cases : {num_train}")
    print(f"Test cases     : {num_test}")
    print(f"Mapping CSV    : {mapping_csv}")
    print(f"Split summary  : {split_summary_csv}")
    if generate_dataset_json:
        print(f"dataset.json   : {out_root / 'dataset.json'}")


if __name__ == "__main__":
    convert_fdg_pet_ct_to_nnunet_input(
        src_root=r"dataset/fdg-pet-ct-lesions/FDG-PET-CT-Lesions",
        out_root=r"dataset/fdg-pet-ct-lesions/nnUNet_raw",
        dataset_id=1,
        dataset_name_suffix="FDGPETCT_autopet3",
        use_symlink=False,      # True = symlink, False = copy
        test_ratio=0.2,         # 20% subjects/cases -> test
        random_seed=42,
        split_by_subject=True,  # strongly recommended
        generate_dataset_json=True,
        verbose=True,
    )
