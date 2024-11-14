import os
import argparse
import nibabel as nib
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def process_session(session_info):
    input_data_path, session_directory, output_data_path = session_info
    session_path = os.path.join(input_data_path, session_directory)
    file_formats = (".nii.gz", ".nii")

    for file in os.listdir(session_path):
        if not file.endswith(file_formats) or not file.startswith("ur_"):
            continue
        
        filepath = os.path.join(session_path, file)
        mri_image = nib.load(filename=filepath)
        mri_data = mri_image.get_fdata()

        cropped_data = mri_data[16:-16, 16:-16, 16:-16]

        output_session_path = os.path.join(output_data_path, session_directory)
        os.makedirs(name=output_session_path, exist_ok=True)
        output_file_path = os.path.join(output_session_path, file)
        output_image = nib.Nifti1Image(cropped_data, affine=mri_image.affine)
        nib.save(output_image, output_file_path)


def main():
    parser = argparse.ArgumentParser("Crop UniRes output images from 196x256x196 to 160x224x160.")
    parser.add_argument("--input", type=str, help="Input dataset path")
    parser.add_argument("--output", type=str, help="Output dataset path")

    args = parser.parse_args()

    input_data_path = args.input
    output_data_path = args.output

    sessions = [(input_data_path, session_directory, output_data_path) 
                for session_directory in os.listdir(input_data_path)]

    with Pool() as pool:
        list(tqdm(pool.imap(process_session, sessions), total=len(sessions)))


if __name__ == "__main__":
    main()
