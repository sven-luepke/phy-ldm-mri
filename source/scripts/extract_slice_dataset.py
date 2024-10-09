import os
import argparse
import nibabel as nib
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool


def process_session(session_info):
    input_3d_data_path, session_directory, output_2d_data_path, slice_count, modalities = session_info
    session_path = os.path.join(input_3d_data_path, session_directory)
    file_formats = (".nii.gz", ".nii")

    for file in os.listdir(session_path):
        if not file.endswith(file_formats) or not file.startswith("ur_"):
            continue

        # file_name_parts = file.split("_")[3:]
        # file_name_parts = [x for x in file_name_parts if not x.startswith("run")]

        # modality_name = "_".join(file_name_parts).split(".")[0]
        
        # if modality_name not in modalities:
        #     continue

        # print(file)
        
        filepath = os.path.join(session_path, file)
        mri_image = nib.load(filename=filepath)
        mri_data = mri_image.get_fdata()

        center_slice = mri_data.shape[2] // 2
        start_slice = center_slice - slice_count // 2

        for slice_index in range(start_slice, start_slice + slice_count):
            slice_session_name = f"{session_directory}_{slice_index}"
            output_session_path = os.path.join(output_2d_data_path, slice_session_name)
            os.makedirs(name=output_session_path, exist_ok=True)

            selected_slice = mri_data[:, :, slice_index]
            selected_slice_3d = selected_slice[:, :, np.newaxis]

            output_file_path = os.path.join(output_session_path, file)
            output_image = nib.Nifti1Image(selected_slice_3d, affine=mri_image.affine)
            nib.save(output_image, output_file_path)


def main():
    parser = argparse.ArgumentParser("Convert samples from a 3D dataset into multiple samples consisting of 2D slices.")
    parser.add_argument("--input", type=str, help="Input 3D dataset path")
    parser.add_argument("--output", type=str, help="Output 2D slice dataset path")
    parser.add_argument("--slices", default=8, type=int, help="Number of axial slices to extract")
    parser.add_argument("--modalities", type=str, help="Modalities to include. By default all modalities are used.")

    args = parser.parse_args()

    modalities = args.modalities.split(",") if args.modalities is not None else None
    input_3d_data_path = args.input
    output_2d_data_path = args.output
    slice_count = args.slices

    sessions = [(input_3d_data_path, session_directory, output_2d_data_path, slice_count, modalities) 
                for session_directory in os.listdir(input_3d_data_path)]

    with Pool() as pool:
        list(tqdm(pool.imap(process_session, sessions), total=len(sessions)))


if __name__ == "__main__":
    main()
