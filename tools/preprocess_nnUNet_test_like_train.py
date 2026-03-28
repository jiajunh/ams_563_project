import os

# Set nnU-Net paths BEFORE importing any nnunetv2 module
os.environ["nnUNet_raw"] = "/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/nnUNet_results"

from batchgenerators.utilities.file_and_folder_operations import (
    join,
    load_json,
    maybe_mkdir_p,
    subfiles,
)
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

def preprocess_test_with_gt_like_train():
    dataset_name = "Dataset001_FDGPETCT_autopet3"
    plans_identifier = "nnUNetPlans"   # or "nnUNetResEncUNetMPlans"
    configuration_name = "3d_fullres"

    print("nnUNet_raw =", nnUNet_raw)
    print("nnUNet_preprocessed =", nnUNet_preprocessed)

    images_ts_dir = join(nnUNet_raw, dataset_name, "imagesTs")
    labels_ts_dir = join(nnUNet_raw, dataset_name, "labelsTs")

    plans_file = join(nnUNet_preprocessed, dataset_name, f"{plans_identifier}.json")

    dataset_json_file_raw = join(nnUNet_raw, dataset_name, "dataset.json")
    dataset_json_file_pp = join(nnUNet_preprocessed, dataset_name, "dataset.json")
    dataset_json_file = dataset_json_file_raw if os.path.isfile(dataset_json_file_raw) else dataset_json_file_pp

    output_dir = join(
        nnUNet_preprocessed,
        dataset_name,
        f"{plans_identifier}_{configuration_name}_test_with_gt"
    )

    if not os.path.isdir(images_ts_dir):
        raise FileNotFoundError(f"imagesTs folder not found: {images_ts_dir}")
    if not os.path.isdir(labels_ts_dir):
        raise FileNotFoundError(f"labelsTs folder not found: {labels_ts_dir}")
    if not os.path.isfile(plans_file):
        raise FileNotFoundError(f"Plans file not found: {plans_file}")
    if not os.path.isfile(dataset_json_file):
        raise FileNotFoundError(f"dataset.json not found: {dataset_json_file}")

    maybe_mkdir_p(output_dir)

    plans = load_json(plans_file)
    dataset_json = load_json(dataset_json_file)

    plans_manager = PlansManager(plans)
    configuration_manager = plans_manager.get_configuration(configuration_name)
    preprocessor = DefaultPreprocessor(verbose=True)

    file_ending = dataset_json["file_ending"]
    num_channels = len(dataset_json["channel_names"])

    case_ids = sorted({
        f[:-len(f"_0000{file_ending}")]
        for f in subfiles(images_ts_dir, suffix=f"_0000{file_ending}", join=False)
    })

    print(f"Found {len(case_ids)} test cases")

    for case_id in case_ids:
        image_files = [
            join(images_ts_dir, f"{case_id}_{c:04d}{file_ending}")
            for c in range(num_channels)
        ]
        seg_file = join(labels_ts_dir, f"{case_id}{file_ending}")
        output_prefix = join(output_dir, case_id)

        missing_images = [f for f in image_files if not os.path.isfile(f)]
        if missing_images:
            print(f"[Skip] {case_id}: missing image files: {missing_images}")
            continue
        if not os.path.isfile(seg_file):
            print(f"[Skip] {case_id}: missing label file: {seg_file}")
            continue

        preprocessor.run_case_save(
            output_filename_truncated=output_prefix,
            image_files=image_files,
            seg_file=seg_file,
            plans_manager=plans_manager,
            configuration_manager=configuration_manager,
            dataset_json=dataset_json,
        )

        print(f"[Saved] {case_id}")


if __name__ == "__main__":
    preprocess_test_with_gt_like_train()