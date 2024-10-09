import os
import json
import random
from typing import Tuple, List, Optional

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from monai.data import CacheDataset
from monai.transforms import (
    Compose,
    LoadImaged,
    SqueezeDimd,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityd,
    ToTensord,
    MapTransform,
    Randomizable
)
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def create_tensorboard_writer(experiment_name: str, log_dir_root: str):
    logdir = os.path.join(log_dir_root, experiment_name)
    os.makedirs(logdir, exist_ok=True)
    experiment_index = len(os.listdir(logdir))
    return SummaryWriter(os.path.join(logdir, str(experiment_index).zfill(3)))


def create_oasis_3_mr_data_split(sessions_json_path: str) -> Tuple[List[str], List[str]]:
    """ Create a train/validation split from a list of MR session IDs in a json file. """

    with open(sessions_json_path, "r") as file:
        mr_session_ids = json.load(file)

    random.seed(42)
    random.shuffle(mr_session_ids)
    return train_test_split(mr_session_ids, test_size=0.1, random_state=420)


def load_oasis_3_mr_meta_df(csv_file_path: str) -> pd.DataFrame:
    """
    Load a dataframe containing meta data of MRI scans including acquisition parameters and modality IDs.
    The raw csv has been created by the collect_acq_params.py script.
    """

    df = pd.read_csv(csv_file_path)

    # add modality column from scan ID
    def scan_id_to_modality(scan_id):
        if "acq-TSE" in scan_id:
            assert "T2w" in scan_id
            return "TSE_T2w"
        else:
            return scan_id.split('_')[-1]  # Default behavior

    df["Modality"] = df["ScanID"].apply(scan_id_to_modality)

    # Mean fill missing values for T1w MPRAGE scans
    df.loc[df["InversionTime"].isna() & (df["Modality"] == "T1w"), "InversionTime"] = df[df["Modality"] == "T1w"]["InversionTime"].mean()

    # fill inversion time values of FLAIR scans
    flair_mask = df["Modality"] == "FLAIR"
    df.loc[flair_mask & df["InversionTime"].isna() & (df["RepetitionTime"] == 5), "InversionTime"] = 1.8
    df.loc[flair_mask & df["InversionTime"].isna() & (df["RepetitionTime"] >= 9), "InversionTime"] = 2.5

    df.fillna(value=0, inplace=True)

    return df


class RandomMask(MapTransform, Randomizable):
    """
    Custom MONAI transform that randomly sets values in a binary mask to zero,
    ensuring at least one value remains 1.
    """
    def __init__(self, keys, prob=0.1, seed=None):
        super().__init__(keys)
        self.prob = prob
        self.seed = seed
        self.set_random_state(seed=self.seed)

    def __call__(self, data):
        # Ensure the seed is reused to maintain randomness behavior
        self.randomize(None)
        d = dict(data)
        
        for key in self.keys:
            mask = d[key]
            
            # Randomly set values to zero based on the specified probability
            random_mask = self.R.random(mask.shape) < self.prob
            modified_mask = np.where(random_mask, 0, mask)

            # Ensure at least one value is 1
            if modified_mask.sum() == 0:
                modified_mask = mask
            
            d[key + "_input"] = modified_mask
        
        return d

    def randomize(self, data=None):
        self.R = np.random.RandomState(self.R.randint(np.iinfo(np.int32).max))


ECHO_TIME_INDEX = 0
REPETITION_TIME_INDEX = 1
INVERSION_TIME_INDEX = 2

T1W_MODALITY_ID = 0
T2W_MODALITY_ID = 1
FLAIR_MODALITY_ID = 2


def create_datasets(
        image_dataset_path: str,
        train_session_ids: List[str],
        val_session_ids: List[str],
        modalities: List[str],
        require_all_modalities: bool,
        meta_df: pd.DataFrame,
        sample_limit: Optional[int] = None
    ) -> Tuple[CacheDataset, CacheDataset]:
    # Create datalists
    train_data_list = []
    val_data_list = []

    for session in tqdm(os.listdir(image_dataset_path), desc="Creating datalist"):
        # remove slice index postfix
        session_id = "_".join(session.split("_")[:-1])

        is_train_session = True
        if session_id in train_session_ids:
            is_train_session = True
        elif session_id in val_session_ids:
            is_train_session = False
        else:
            continue

        session_path = os.path.join(image_dataset_path, session)

        data_dict = {
            "images": [],
            "acq_params": [],
            "acq_params_norm": [],
            "modality_id": [],
        }

        meta_df = meta_df[meta_df["Modality"].isin(modalities)]

        image_count = 0
        has_t1w = False
        has_t2w = False
        has_flair = False

        for image_file in os.listdir(session_path):
            if not image_file.endswith(".nii.gz"):
                continue

            if "mask" in image_file:
                # ignore brain masks
                continue

            scan_id = image_file.split(".")[0][3:]  # remove unires "ur_" prefix
            scan_meta_data = meta_df[meta_df["ScanID"] == scan_id]
            if len(scan_meta_data) == 0:
                continue

            data_dict["images"].append(os.path.join(session_path, image_file))

            # import nibabel as nib
            # d = nib.load(os.path.join(session_path, image_file)).get_fdata()
            # print(d.shape)

            # acquisition parameters
            acq_params = [0] * 3
            acq_params[ECHO_TIME_INDEX] = scan_meta_data["EchoTime"].iloc[0]
            acq_params[REPETITION_TIME_INDEX] = scan_meta_data["RepetitionTime"].iloc[0]
            acq_params[INVERSION_TIME_INDEX] = scan_meta_data["InversionTime"].iloc[0]
            data_dict["acq_params"].append(acq_params)

            # normalized acqusition paramters
            acq_params_norm = [0] * 3
            acq_params_norm[ECHO_TIME_INDEX] = (
                (acq_params[ECHO_TIME_INDEX] - meta_df["EchoTime"].mean()) / meta_df["EchoTime"].std()
            )
            acq_params_norm[REPETITION_TIME_INDEX] = (
                (acq_params[REPETITION_TIME_INDEX] - meta_df["RepetitionTime"].mean()) / meta_df["RepetitionTime"].std()
            )
            acq_params_norm[INVERSION_TIME_INDEX] = (
                (acq_params[INVERSION_TIME_INDEX] - meta_df["InversionTime"].mean()) / meta_df["InversionTime"].std()
            )
            data_dict["acq_params_norm"].append(acq_params_norm)

            modality_name_to_id = {
                "T1w": T1W_MODALITY_ID,
                "T2w": T2W_MODALITY_ID,
                "FLAIR": FLAIR_MODALITY_ID,
            }
            scan_modality = scan_meta_data["Modality"].iloc[0]
            assert scan_modality in modality_name_to_id
            modality_id = modality_name_to_id[scan_modality]
            data_dict["modality_id"].append(modality_id)

            if modality_id == T1W_MODALITY_ID:
                has_t1w = True
            elif modality_id == T2W_MODALITY_ID:
                has_t2w = True
            elif modality_id == FLAIR_MODALITY_ID:
                has_flair = True

            image_count += 1

        data_dict["image_mask"] = [1] * image_count

        if require_all_modalities:
            if not (has_t1w and has_t2w and has_flair):
                continue
        
        if is_train_session:
            train_data_list.append(data_dict)
        else:
            val_data_list.append(data_dict)

        if sample_limit is not None and len(train_data_list) + len(val_data_list) > sample_limit:
            break

    print(f"Training dataset size = {len(train_data_list)}")
    print(f"Validation dataset size = {len(val_data_list)}")

    # transforms
    transform = Compose(transforms=[
        LoadImaged(keys="images", ensure_channel_first=True),
        SqueezeDimd(keys="images", dim=3),
        ToTensord(keys=["acq_params", "acq_params_norm", "modality_id", "image_mask"]),
        RandomMask(keys="image_mask", prob=0.1),
        ToTensord(keys=["image_mask_input"]),
        #ScaleIntensityRangePercentilesd(keys="images", lower=0, upper=99.5, b_min=0, b_max=1, clip=False, channel_wise=False)
    ])

    #train_data_list = train_data_list[2:3]

    random.seed(42)
    random.shuffle(train_data_list)
    random.shuffle(val_data_list)

    train_dataset = CacheDataset(data=train_data_list, transform=transform, cache_rate=1)
    val_dataset = CacheDataset(data=val_data_list, transform=transform, cache_rate=1)

    return train_dataset, val_dataset


def oasis_multimodal_collate_fn(data):
    """
    Pad data for missing modalities with zeros.
    """

    #max_modality_count = 0
    #for i in range(len(data)):
    #    max_modality_count = max(data[i]["images"].shape[0], max_modality_count)
    max_modality_count = 3

    batch = dict()
    for i in range(len(data)):
        for key, value in data[i].items():
            # pad and add batch dim
            if key == "modality_id":
                pad_value = -1
            else:
                pad_value = 0
            padded_data = F.pad(value, (0, 0) * (value.dim() - 1) + (0, max_modality_count - value.shape[0]), value=pad_value).unsqueeze(0)
            #print(padded_data.shape)
            if i == 0:
                batch[key] = [padded_data]
            else:
                batch[key].append(padded_data)

    for key, value in batch.items():
        batch[key] = torch.concat(value, dim=0)

    return batch


def kl_div(z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
        return torch.sum(kl_loss) / kl_loss.shape[0]