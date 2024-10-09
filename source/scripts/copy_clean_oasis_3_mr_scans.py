import argparse
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def main():
    parser = argparse.ArgumentParser("Copy MR scans filtered based on MR sessions.")
    parser.add_argument("--acq_path", type=str)
    parser.add_argument("--src_scans_path", type=str)
    parser.add_argument("--dst_scans_path", type=str)

    args = parser.parse_args()

    acq_path = args.acq_path
    src_scans_path = args.src_scans_path
    dst_scans_path = args.dst_scans_path

    # Precompute session paths
    session_paths = [(session, os.path.join(acq_path, session),
                      os.path.join(src_scans_path, session),
                      os.path.join(dst_scans_path, session))
                     for session in os.listdir(acq_path)]

    # Process each session in parallel
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        list(tqdm(executor.map(process_session, session_paths), total=len(session_paths)))

def process_session(session_paths):
    session, session_acq_params_path, src_session_path, dst_session_path = session_paths

    if not os.path.exists(src_session_path):
        return  # Skip if source session path doesn't exist

    os.makedirs(dst_session_path, exist_ok=True)
    scan_names = set(scan_name.split(".")[0] for scan_name in os.listdir(session_acq_params_path))

    for current_root, _, files in os.walk(src_session_path):
        for file in files:
            scan_name = file.split(".")[0]
            if scan_name in scan_names:
                src_file_path = os.path.join(current_root, file)
                dst_file_path = os.path.join(dst_session_path, file)
                shutil.copy(src_file_path, dst_file_path)

if __name__ == "__main__":
    main()
