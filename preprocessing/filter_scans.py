import argparse
import os
import shutil

from tqdm import tqdm
import pandas as pd


def main():
    parser = argparse.ArgumentParser("Copy the relevant scans from the OASIS-3 dataset into a flat directory structure.")
    parser.add_argument("--acq_params_csv", type=str)
    parser.add_argument("--src_scans_path", type=str)
    parser.add_argument("--dst_scans_path", type=str)

    # parse cmd args
    args = parser.parse_args()
    src_scans_path = args.src_scans_path
    dst_scans_path = args.dst_scans_path

    scan_ids = set(pd.read_csv(args.acq_params_csv)["ScanID"].to_list())

    os.makedirs(dst_scans_path, exist_ok=True)
    for experiment_dir in tqdm(os.listdir(src_scans_path)):
        new_experiment_dir = os.path.join(dst_scans_path, experiment_dir)
        
        experiment_path = os.path.join(src_scans_path, experiment_dir)
        for current_root, _, files in os.walk(experiment_path):
            for file in files:
                if not file.endswith(".nii.gz"):
                    continue

                scan_id = file.split(".")[0]
                if scan_id not in scan_ids:
                    continue
                
                os.makedirs(new_experiment_dir, exist_ok=True)
                src_file_path = os.path.join(current_root, file)
                dst_file_path = os.path.join(new_experiment_dir, file)
                shutil.copy(src_file_path, dst_file_path)


if __name__ == "__main__":
    main()
