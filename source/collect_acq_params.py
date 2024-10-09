import os
import argparse
import json

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser("Load MRI acqusition parameters from multiple json files in BIDS format into a single csv table.")
    parser.add_argument("--data", type=str, help="Dataset path", required=True)
    parser.add_argument("--out", type=str, default="acq_params.csv", help="Output file path.")

    args = parser.parse_args()
    dataset_path = args.data

    scan_id_list = []
    echo_time_list = []
    repetition_time_list = []
    inversion_time_list = []
    field_strength_list = []

    for session in tqdm(os.listdir(dataset_path)):
        session_path = os.path.join(dataset_path, session)

        for current_root, dirs, files in os.walk(session_path): 

            for file in files:
                if not file.endswith(".json"):
                    continue

                file_path = os.path.join(current_root, file)
                json_data = json.load(open(file_path))

                scan_id = file.split(".")[0]
                echo_time = json_data["EchoTime"]
                repetition_time = json_data["RepetitionTime"]
                inversion_time = json_data.get("InversionTime")
                field_strength = json_data.get("MagneticFieldStrength")

                scan_id_list.append(scan_id)
                echo_time_list.append(echo_time)
                repetition_time_list.append(repetition_time)
                inversion_time_list.append(inversion_time)
                field_strength_list.append(field_strength)

    data_dict = {
        "ScanID": scan_id_list,
        "EchoTime": echo_time_list,
        "RepetitionTime": repetition_time_list,
        "InversionTime": inversion_time_list,
        "FieldStrength": field_strength_list,
    }
    df = pd.DataFrame(data=data_dict)

    df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()