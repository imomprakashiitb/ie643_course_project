from typing import Optional
import torch
from data import BrainDataset  # Updated to use new BrainDataset

class DataDescriptor:
    def __init__(self, n_workers=2, batch_size=16, **kwargs):
        self.n_workers = n_workers
        self.batch_size = batch_size
        self.dataset_cache = {}

    def get_dataset(self, split: str):
        raise NotImplementedError("get_dataset needs to be overridden in a subclass.")

    def get_dataset_(self, split: str, cache=True, force=False):
        if split not in self.dataset_cache or force:
            dataset = self.get_dataset(split)
            if cache:
                self.dataset_cache[split] = dataset
            return dataset
        else:
            return self.dataset_cache[split]

    def get_dataloader(self, split: str):
        dataset = self.get_dataset_(split, cache=True)
        shuffle = True if split == "train" else False
        drop_last = False if len(dataset) < self.batch_size else True
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=shuffle,
                                                 drop_last=drop_last,
                                                 num_workers=self.n_workers,
                                                 pin_memory=True)  # Added for GPU efficiency
        return dataloader

class BrainAEDataDescriptor(DataDescriptor):
    """
    Modified: For mixed healthy + unhealthy training
    - Uses ALL slices from ALL patients (no filtering)
    - n_train_patients = total patients for train (None = all)
    - n_val_patients = total patients for val (None = all)
    - Supports slice_percentage for random slice selection
    """
    def __init__(self, dataset="ixi_braTS",
                 n_train_patients: Optional[int] = None,  # Total patients for train
                 n_val_patients: Optional[int] = None,     # Total patients for val
                 seed: int = 0,
                 data_path: str = "/content/drive/MyDrive/ie643_course_project_24M1644/dae_input_data",
                 slice_percentage: float = 0.5,  # Default to 50% slices per MRI
                 **kwargs):
        super().__init__(n_workers=2, batch_size=kwargs.get('batch_size', 16))
        self.seed = seed
        self.dataset = dataset
        self.n_train_patients = n_train_patients  # Now means TOTAL patients
        self.n_val_patients = n_val_patients      # Now means TOTAL patients
        self.data_path = data_path
        self.slice_percentage = slice_percentage  # Added to align with train_dae_colab.py

    def get_dataset(self, split: str):
        """
        Modified: Uses new BrainDataset interface for mixed training
        - split: 'train' or 'val'
        - n_patients: Total patients to load (None = all patients in split)
        - skip_condition=None: Loads ALL slices (healthy + unhealthy)
        - slice_percentage: Randomly selects percentage of slices per patient
        """
        assert split in ["train", "val"]

        # Determine number of patients for this split
        n_patients = (self.n_train_patients if split == "train"
                     else self.n_val_patients)

        dataset = BrainDataset(
            split=split,
            dataset=self.dataset,
            n_patients=n_patients,  # Total patients (None = all)
            seed=self.seed if split == "train" else 0,  # Fixed seed for val
            data_path=self.data_path,
            slice_percentage=self.slice_percentage  # Pass slice_percentage to BrainDataset
        )

        print(f"{split.upper()} dataset: {len(dataset)} slices loaded")
        return dataset