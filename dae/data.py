from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np

class PatientDataset(Dataset):
    """
    Dataset class representing a collection of slices from a single patient.
    Modified: No skip_condition by default for mixed training, supports random slice selection.
    """
    def __init__(self, patient_dir: Path, process_fun=None, id=None, skip_condition=None, slice_percentage: float = 1.0):
        self.patient_dir = patient_dir
        # Sort slices by slice number for consistent ordering
        self.slice_paths = sorted(list(patient_dir.iterdir()), key=lambda x: int(x.name[6:-4]))
        self.process = process_fun
        self.skip_condition = skip_condition
        self.id = id
        self.slice_percentage = slice_percentage
        self.len = len(self.slice_paths)
        self.idx_map = {x: x for x in range(self.len)}

        # Apply random slice selection based on percentage
        if self.slice_percentage < 1.0:
            selected_indices = random.sample(range(self.len), int(self.len * self.slice_percentage))
            self.len = len(selected_indices)
            self.idx_map = {x: selected_indices[x] for x in range(self.len)}

        # Apply skip_condition if provided (disabled for mixed training)
        if self.skip_condition is not None:
            valid_indices = []
            for idx in range(self.len):
                with np.load(self.slice_paths[self.idx_map[idx]]) as data:
                    if self.process is not None:
                        data = self.process(**data)
                    if not skip_condition(data):
                        valid_indices.append(idx)
            self.len = len(valid_indices)
            self.idx_map = {x: self.idx_map[valid_indices[x]] for x in range(self.len)}

    def __getitem__(self, idx):
        idx = self.idx_map[idx]
        data = np.load(self.slice_paths[idx])
        if self.process is not None:
            data = self.process(**data)
        return data

    def __len__(self):
        return self.len

class BrainDataset(Dataset):
    """
    Dataset class for mixed healthy + unhealthy slices training.
    Modified: Loads ALL slices from ALL patients without filtering, supports slice percentage.
    """
    def __init__(self, dataset="ixi_braTS", split="val", n_patients=None,
                 seed=0, data_path="/content/drive/MyDrive/ie643_course_project_24M1644/dae_input_data", slice_percentage: float = 1.0):
        """
        Args:
            n_patients: Number of patients to use (None = all patients)
            seed: Random seed for shuffling patients
            slice_percentage: Percentage of slices to randomly select per patient (default 1.0)
        """
        self.rng = random.Random(seed)
        assert split in ["train", "val", "test"]

        path = Path(data_path) / split

        # Verify path exists
        assert path.exists(), f"Path does not exist: {path}"

        # Get all patient directories and shuffle for randomness
        patient_dirs = sorted(list(path.iterdir()))
        self.rng.shuffle(patient_dirs)

        # Limit number of patients if specified
        if n_patients is not None:
            patient_dirs = patient_dirs[:n_patients]

        print(f"Loading {len(patient_dirs)} patients from {split} split")

        # For DAE, we use x (T2 slice) as both input and target, y is not used for reconstruction
        def process(x, y):
            # Extract T2 slice (x) and tumor mask (y) as single-channel tensors
            y = y > 0.5
            return torch.from_numpy(x[0]).float(), torch.from_numpy(y[0]).float()

        # Create PatientDataset for ALL patients with NO skip_condition and custom slice_percentage
        self.patient_datasets = [
            PatientDataset(patient_dir, process_fun=process, id=i, skip_condition=None, slice_percentage=slice_percentage)
            for i, patient_dir in enumerate(patient_dirs)
        ]

        # Combine all patient datasets into one
        self.dataset = ConcatDataset(self.patient_datasets)
        print(f"Total slices loaded: {len(self.dataset)}")
        print(f"Average slices per patient: {len(self.dataset) / len(patient_dirs):.1f}")

    def __getitem__(self, idx):
        x, gt = self.dataset[idx]
        return x, gt  # x: T2 slice, gt: tumor mask (used as clean target in DAE)

    def __len__(self):
        return len(self.dataset)

    def get_all_slices(self):
        """Return a dict mapping patient IDs to their slice indices."""
        slice_dict = {}
        start_idx = 0
        for i, patient_dataset in enumerate(self.patient_datasets):
            slice_dict[i] = list(range(start_idx, start_idx + len(patient_dataset)))
            start_idx += len(patient_dataset)
        return slice_dict

    def update_slice_selection(self, selected_slices):
        """Update the dataset with a new selection of slices per patient."""
        new_datasets = []
        for patient_id, indices in selected_slices.items():
            patient_dataset = self.patient_datasets[patient_id]
            patient_dataset.idx_map = {i: patient_dataset.idx_map[idx] for i, idx in enumerate(indices)}
            patient_dataset.len = len(indices)
            new_datasets.append(patient_dataset)
        self.dataset.datasets = new_datasets
        self.dataset.cumulative_sizes = [len(d) for d in new_datasets]
        for i in range(1, len(self.dataset.cumulative_sizes)):
            self.dataset.cumulative_sizes[i] += self.dataset.cumulative_sizes[i-1]