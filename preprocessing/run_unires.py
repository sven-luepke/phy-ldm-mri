import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser("Run UniRes on the dataset")
    parser.add_argument("--data", type=str)
    args = parser.parse_args()

    dataset_path = args.data

    session_dirs = sorted(os.listdir(dataset_path))

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