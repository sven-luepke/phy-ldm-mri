import argparse
import os
import subprocess
import math


def split_list_evenly(lst, n):
    sublist_size =math. ceil(len(lst) / n)
    return [lst[i:i + sublist_size] for i in range(0, len(lst), sublist_size)]


def main():
    parser = argparse.ArgumentParser("Run UniRes on the dataset")
    parser.add_argument("--data", type=str)
    parser.add_argument("--split_count", type=int, default=1)
    parser.add_argument("--split_index", type=int, default=0)
    args = parser.parse_args()

    dataset_path = args.data

    session_dirs = sorted(os.listdir(dataset_path))

    session_dirs = split_list_evenly(session_dirs, args.split_count)[args.split_index]

    for session in session_dirs:
        session_path = os.path.join(dataset_path, session)

        session_files = os.listdir(session_path)
        unires_files = [s for s in session_files if s.startswith("ur_")]
        if len(unires_files) > 0:
            print(f"Skipping session {session}")
            continue

        raw_mri_files = set([s for s in session_files if (not s.startswith("ur_") and s.endswith("nii.gz"))])
        
        cmd = ["unires", "--linear", "--common_output"]

        for file in raw_mri_files:
            if "T1w" in file:
                file_path = os.path.join(session_path, file)
                cmd += [file_path]
                raw_mri_files.remove(file)
                break

        for file in raw_mri_files:
             file_path = os.path.join(session_path, file)
             cmd += [file_path]

        subprocess.run(cmd, text=True, check=True)


if __name__ == "__main__":
    main()