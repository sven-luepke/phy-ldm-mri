import argparse
import os
import json
import math
import shutil


def main():
    parser = argparse.ArgumentParser("Write all clean MRI acqusition parameters to a new directory.")
    parser.add_argument("--input", type=str, help="Dataset directory containing the acquisition parameters")
    parser.add_argument("--output", type=str, help="Output data path")

    with open("oasis_3_mr_session_ids_clean.csv", 'r') as filename:
        data = filename.readlines()
    valid_sessions = set([line.strip() for line in data])
    
    args = parser.parse_args()
    dataset_path = args.input
    output_path = args.output

    os.makedirs(name=output_path, exist_ok=True)

    valid_file_count = 0
    multi_run_scan_count = 0
    echo_scans = 0
    for current_root, dirs, files in os.walk(dataset_path):
        for filename in files:
            if not filename.endswith(".json") or "angio" in filename or "":
                # skip MRA scans
                continue

            if "echo" in filename:
                print("Skip echo")
                print(filename)
                echo_scans += 1
                continue
            
            session_id = current_root.split("/")[-3]
            if session_id not in valid_sessions:
                # skip session that we could not perform pre-processing on
                continue

            file_path = os.path.join(current_root, filename)
            json_data = json.load(open(file_path))

            

            if "_run-0" in filename:
                multi_run_scan_count += 1
                #print(file)

            if "T1w" in filename:
                if "SeriesDescription" not in json_data:
                    print("Missing Series Description:")
                    print(filename)
                    continue
                if "mpr" not in json_data["SeriesDescription"].lower():
                    print("Found non-MPRAGE T1w scan:")
                    print(filename)
                    continue

            if "hippocampus" in filename:
                #print("Skipping hippocampus scan:")
                #print(filename)
                continue

            if "RepetitionTime" not in json_data:
                # Skip that file
                #print("Missing Repetition time in file: ")
                #print(filename)
                continue

            if "EchoTime" not in json_data:
                # Skip that file
                #print("Missing Echo time in file: ")
                #print(filename)
                continue

            if "FlipAngle" not in json_data:
                #print("Missing Flip angle in file: ")
                #print(filename)
                continue

            valid_file_count += 1

            file_out_path = os.path.join(output_path, session_id)
            os.makedirs(name=file_out_path, exist_ok=True)
            
            shutil.copy(src=file_path, dst=os.path.join(file_out_path, filename))

    for session in os.listdir(output_path):
        session_dir = os.path.join(output_path, session)
        session_files = set(os.listdir(session_dir))

        modalities = ["acq-TSE", "FLAIR", "T1w", "T2w"]
        for modality in modalities:

            duplicate_runs = sorted([s for s in session_files if ("_run-0" in s and modality in s)])
            for i in range(0, len(duplicate_runs) - 1):
                path = os.path.join(session_dir, duplicate_runs[i])
                if not os.path.exists(path=path):
                    continue
                os.remove(path=path)
                valid_file_count -= 1           

    print(valid_file_count)
    print(echo_scans)


if __name__ == "__main__":
    main()