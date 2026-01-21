import os
import random

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from sleepfm.utils import load_data


class DiagnosisFinetuneFullCOXPHWithDemoDataset(Dataset):
    def __init__(
        self,
        config,
        channel_groups,
        hdf5_paths=None,
        demo_labels_path=None,
        split="train",
    ):
        self.config = config
        self.channel_groups = channel_groups
        self.max_channels = self.config["max_channels"]

        # --- Load demographic features ---
        if not demo_labels_path:
            demo_labels_path = config["demo_labels_path"]

        demo_labels_df = pd.read_csv(demo_labels_path)
        demo_labels_df = demo_labels_df.set_index("Study ID")
        study_ids = set(demo_labels_df.index)

        is_event_df = pd.read_csv(
            os.path.join(self.config["labels_path"], "is_event.csv")
        )
        event_time_df = pd.read_csv(
            os.path.join(self.config["labels_path"], "time_to_event.csv")
        )

        is_event_df = is_event_df.set_index("Study ID")
        event_time_df = event_time_df.set_index("Study ID")

        # --- Resolve HDF5 paths (explicit precedence) ---
        if hdf5_paths:
            # Use provided paths directly
            hdf5_paths = [f for f in hdf5_paths if os.path.exists(f)]
        else:
            # Load from split file
            split_paths = load_data(config["split_path"])[split]
            hdf5_paths = [f for f in split_paths if os.path.exists(f)]

        # Filter by available demo labels
        hdf5_paths = [
            f for f in hdf5_paths if os.path.basename(f).split(".")[0] in study_ids
        ]

        # Optional truncation
        if config.get("max_files"):
            hdf5_paths = hdf5_paths[: config["max_files"]]

        labels_dict = {}
        # Loop over each study_id
        for study_id in tqdm(study_ids):
            # Extract the row as a whole for both dataframes (faster than iterating over columns)
            is_event_row = list(is_event_df.loc[study_id].values)
            event_time_row = list(event_time_df.loc[study_id].values)
            demo_feats = list(demo_labels_df.loc[study_id].values)

            # values = [[event_time, is_event] for is_event, event_time in zip(is_event_row, event_time_row)]
            labels_dict[study_id] = {
                "is_event": is_event_row,
                "event_time": event_time_row,
                "demo_feats": demo_feats,
            }

        # --- Build index map ---
        self.index_map = [
            (path, labels_dict[os.path.basename(path).split(".")[0]])
            for path in hdf5_paths
        ]

        print(f"Number of files in {split} set: {len(hdf5_paths)}")
        print(f"Number of files to be processed in {split} set: {len(self.index_map)}")

        self.total_len = len(self.index_map)
        self.max_seq_len = config["model_params"]["max_seq_length"]

        if self.total_len == 0:
            raise ValueError(f"No valid HDF5 files found for split='{split}'.")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        hdf5_path, tte_event = self.index_map[idx]

        event_time = tte_event["event_time"]
        is_event = tte_event["is_event"]
        demo_feats = tte_event["demo_feats"]

        x_data = []
        with h5py.File(hdf5_path, "r") as hf:
            dset_names = []
            for dset_name in hf.keys():
                if (
                    isinstance(hf[dset_name], h5py.Dataset)
                    and dset_name in self.config["modality_types"]
                ):
                    dset_names.append(dset_name)

            random.shuffle(dset_names)
            for dataset_name in dset_names:
                x_data.append(hf[dataset_name][:])

        if not x_data:
            # Skip this data point if x_data is empty
            return self.__getitem__((idx + 1) % self.total_len)

        # Convert x_data list to a single numpy array
        x_data = np.array(x_data)

        # Convert x_data to tensor
        x_data = torch.tensor(x_data, dtype=torch.float32)

        event_time = torch.tensor(event_time, dtype=torch.float32)
        is_event = torch.tensor(is_event)

        demo_feats = torch.tensor(demo_feats, dtype=torch.float32)

        return (
            x_data,
            event_time,
            is_event,
            demo_feats,
            self.max_channels,
            self.max_seq_len,
            hdf5_path,
        )


def diagnosis_finetune_full_coxph_with_demo_collate_fn(batch):
    (
        x_data,
        event_time,
        is_event,
        demo_feats,
        max_channels_list,
        max_seq_len_list,
        hdf5_path_list,
    ) = zip(*batch)

    num_channels = max(max_channels_list)

    if max_seq_len_list[0] == None:
        max_seq_len = max([item.size(1) for item in x_data])
    else:
        max_seq_len = max_seq_len_list[0]

    padded_x_data = []
    padded_mask = []
    for item in x_data:
        c, s, e = item.size()
        c = min(c, num_channels)
        s = min(s, max_seq_len)  # Ensure the sequence length doesn't exceed max_seq_len

        # Create a padded tensor and a mask tensor
        padded_item = torch.zeros((num_channels, max_seq_len, e))
        mask = torch.ones((num_channels, max_seq_len))

        # Copy the actual data to the padded tensor and set the mask for real data
        padded_item[:c, :s, :e] = item[:c, :s, :e]
        mask[:c, :s] = 0  # 0 for real data, 1 for padding

        padded_x_data.append(padded_item)
        padded_mask.append(mask)

    # Stack all tensors into a batch
    x_data = torch.stack(padded_x_data)
    event_time = torch.stack(event_time)
    is_event = torch.stack(is_event)
    demo_feats = torch.stack(demo_feats)
    padded_mask = torch.stack(padded_mask)

    return x_data, event_time, is_event, demo_feats, padded_mask, hdf5_path_list
