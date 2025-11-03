import torch
import random
from pathlib import Path
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torch.nn.functional as F

def normalise_percentile_t2(volume):
    if volume.ndim == 3:
        volume = volume.unsqueeze(1)
    v_ = volume[:, 0, :, :].reshape(-1)
    v_ = v_[v_ > 0]
    if v_.numel() == 0:
        return volume
    p_99 = torch.quantile(v_, 0.99)
    volume[:, 0, :, :] /= p_99
    return volume

def process_patient_t2(path, target_path):
    t2_file = path / f"{path.name}_t2.nii.gz"
    seg_file = path / f"{path.name}_seg.nii.gz"
    if not t2_file.exists() or not seg_file.exists():
        return
    try:
        t2 = nib.load(t2_file).get_fdata()
        labels = nib.load(seg_file).get_fdata()
    except Exception:
        return

    t2 = torch.from_numpy(t2).unsqueeze(0).unsqueeze(0)
    labels = torch.from_numpy((labels > 0.5).astype(np.float32)).unsqueeze(0).unsqueeze(0)
    t2 = normalise_percentile_t2(t2)

    patient_dir = target_path / f"patient_{path.name}"
    try:
        patient_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        return

    mean_vol = t2[0, 0]
    sum_dim2 = (mean_vol.sum(dim=0).sum(dim=0) > 0.1).int()
    fs_dim2 = sum_dim2.argmax().item()
    ls_dim2 = mean_vol.shape[2] - sum_dim2.flip(dims=[0]).argmax().item()

    if fs_dim2 >= ls_dim2:
        return

    for slice_idx in range(fs_dim2, ls_dim2):
        slice_t2 = t2[:, :, :, :, slice_idx]
        slice_lbl = labels[:, :, :, :, slice_idx]
        low_res_x = F.interpolate(slice_t2, mode="bilinear", size=(128, 128))
        low_res_y = F.interpolate(slice_lbl, mode="bilinear", size=(128, 128))
        try:
            output_file = patient_dir / f"slice_{slice_idx}.npz"
            np.savez_compressed(output_file, x=low_res_x.numpy(), y=low_res_y.numpy())
        except Exception:
            pass

def preprocess(datapath: Path):
    all_imgs = sorted([p for p in datapath.iterdir() if p.is_dir()])
    if not all_imgs:
        return

    valid_imgs = [p for p in all_imgs if (p / f"{p.name}_t2.nii.gz").exists() and (p / f"{p.name}_seg.nii.gz").exists()]
    if not valid_imgs:
        return

    splits_path = Path("E:/Study/IIT_Bombay/SEM_3/IE643_DeepLearning/Project/ie643_project/Data/dae_input")
    for split in ["train", "val", "test"]:
        (splits_path / split).mkdir(parents=True, exist_ok=True)
    indices = list(range(len(valid_imgs)))
    random.seed(0)
    random.shuffle(indices)
    n_train = int(len(indices) * 0.75)
    n_val = int(len(indices) * 0.05)
    n_test = len(indices) - n_train - n_val
    split_indices = {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }
    for split in ["train", "val", "test"]:
        try:
            with open(splits_path / split / "scans.csv", "w") as f:
                f.write("\n".join([valid_imgs[idx].name for idx in split_indices[split]]))
        except Exception:
            pass

    for split in ["train", "val", "test"]:
        scans_csv = splits_path / split / "scans.csv"
        try:
            with open(scans_csv, "r") as f:
                lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        except FileNotFoundError:
            continue
        paths = [datapath / x for x in lines]
        target_base = Path(f"E:/Study/IIT_Bombay/SEM_3/IE643_DeepLearning/Project/ie643_project/Data/dae_input/{split}")
        try:
            target_base.mkdir(parents=True, exist_ok=True)
        except Exception:
            continue
        for source_path in tqdm(paths, desc=f"Processing {split}"):
            process_patient_t2(source_path, target_base)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", required=True, type=str,
                        help="path to Brats2021/IXI Training Data directory")
    args = parser.parse_args()
    datapath = Path(args.source)
    preprocess(datapath)