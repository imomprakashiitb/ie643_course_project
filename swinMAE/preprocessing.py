import os
import glob
import argparse
import numpy as np
import h5py
import nibabel as nib
from tqdm import tqdm
import csv

def preprocess_volume(image_path, mask_path, seg_src_path, output_h5_path, seg_h5_path,
                      target_size=224, start_slice=30, end_slice=10):
    """
    seg_src_path: path to the existing seg .nii/.nii.gz in input folder (may be same as mask_path or separate).
    We create output_h5 (image+label) using mask_path, and seg_h5 using seg_src_path (processed same way).
    """
    print(f"  -> preprocess_volume called for:\n     image: {image_path}\n     mask:  {mask_path}\n     segsrc:{seg_src_path}\n     out:   {output_h5_path}\n     seg:   {seg_h5_path}")

    img_nii = nib.load(image_path)
    img_data = img_nii.get_fdata()  # (H, W, D)
    affine = img_nii.affine

    mask_nii = nib.load(mask_path)
    mask_data = mask_nii.get_fdata()

    # seg_src_path may be None (no separate seg provided) — in that case we use mask_data for seg_h5
    seg_data = None
    if seg_src_path is not None:
        seg_nii = nib.load(seg_src_path)
        seg_data = seg_nii.get_fdata()
        if seg_data.shape[2] != img_data.shape[2]:
            raise ValueError(f"Depth mismatch between image and seg file: {img_data.shape[2]} vs {seg_data.shape[2]}")

    if img_data.shape[2] != mask_data.shape[2]:
        raise ValueError(f"Depth mismatch: image has {img_data.shape[2]} but mask has {mask_data.shape[2]} slices")

    nslices = img_data.shape[2] - end_slice
    effective_slices = range(start_slice, nslices)
    n_slices = len(effective_slices)

    img_shape = (1, target_size, target_size, n_slices)
    label_shape = img_shape

    # Create main GT file (image + label)
    with h5py.File(output_h5_path, 'w') as h5f:
        img_ds = h5f.create_dataset('image', img_shape, dtype=np.float32,
                                    chunks=(1, target_size, target_size, 1))
        label_ds = h5f.create_dataset('label', label_shape, dtype=np.uint8,
                                      chunks=(1, target_size, target_size, 1))

        for i, slicenum in enumerate(tqdm(effective_slices, desc=f"    slices")):
            img_slice = img_data[:, :, slicenum]
            mask_slice = mask_data[:, :, slicenum]

            # Rotate 90° clockwise
            img_slice = np.rot90(img_slice, k=-1)
            mask_slice = np.rot90(mask_slice, k=-1)

            # Normalize to [0, 1]
            if img_slice.max() - img_slice.min() > 0:
                img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            else:
                img_slice = np.zeros_like(img_slice)

            # Binarize mask
            mask_slice = (mask_slice > 0).astype(np.uint8)

            # Pad to target size
            h, w = img_slice.shape
            if h > target_size or w > target_size:
                raise ValueError(f"Slice size ({h},{w}) larger than target_size {target_size}.")
            pad_h_before = (target_size - h) // 2
            pad_h_after = target_size - h - pad_h_before
            pad_w_before = (target_size - w) // 2
            pad_w_after = target_size - w - pad_w_before

            img_slice = np.pad(img_slice, ((pad_h_before, pad_h_after),
                                           (pad_w_before, pad_w_after)), mode='constant')
            mask_slice = np.pad(mask_slice, ((pad_h_before, pad_h_after),
                                             (pad_w_before, pad_w_after)), mode='constant')

            img_ds[0, :, :, i] = img_slice
            label_ds[0, :, :, i] = mask_slice

    np.savez(output_h5_path.replace('.h5', '_meta.npz'), affine=affine)
    print(f"  Saved main file: {output_h5_path}")

    # Create seg file using seg_data if available, otherwise fallback to mask_data
    seg_source = seg_data if seg_data is not None else mask_data
    with h5py.File(seg_h5_path, 'w') as segf:
        pred_ds = segf.create_dataset('prediction', label_shape, dtype=np.uint8,
                                     chunks=(1, target_size, target_size, 1))
        for i, slicenum in enumerate(effective_slices):
            mask_slice = seg_source[:, :, slicenum]
            mask_slice = np.rot90(mask_slice, k=-1)
            mask_slice = (mask_slice > 0).astype(np.uint8)
            h, w = mask_slice.shape
            pad_h_before = (target_size - h) // 2
            pad_h_after = target_size - h - pad_h_before
            pad_w_before = (target_size - w) // 2
            pad_w_after = target_size - w - pad_w_before
            mask_slice = np.pad(mask_slice, ((pad_h_before, pad_h_after),
                                             (pad_w_before, pad_w_after)), mode='constant')
            pred_ds[0, :, :, i] = mask_slice
        segf.attrs['description'] = 'Segmentation H5 created from existing seg/ mask file.'
    np.savez(seg_h5_path.replace('.h5', '_meta.npz'), affine=affine)
    print(f"  Saved seg file: {seg_h5_path}")


def find_image_mask_seg_in_folder(folder):
    """
    Detect image, mask, and seg files in the sample folder.
    Returns (image_path, mask_path, seg_path_or_None)
    """
    candidates = sorted(glob.glob(os.path.join(folder, '*.nii')) +
                        glob.glob(os.path.join(folder, '*.nii.gz')))
    if not candidates:
        return None, None, None

    # classify names
    image_candidates = []
    mask_candidates = []
    seg_candidates = []
    for c in candidates:
        name = os.path.basename(c).lower()
        if 't2' in name or 't1' in name or 'flair' in name:
            image_candidates.append(c)
        elif '_seg' in name or name.endswith('seg.nii') or 'seg.' in name:
            seg_candidates.append(c)
        elif '_mask' in name or 'mask' in name or 'gt' in name or 'ground' in name:
            mask_candidates.append(c)
        else:
            # fallback: treat as image if nothing else found
            image_candidates.append(c)

    # deduce image
    image_path = None
    if image_candidates:
        # prefer T2 explicitly
        t2 = [c for c in image_candidates if 't2' in os.path.basename(c).lower()]
        image_path = t2[0] if t2 else image_candidates[0]

    # deduce mask (ground truth)
    mask_path = mask_candidates[0] if mask_candidates else None

    # deduce seg (existing segmentation file)
    seg_path = None
    # prefer files explicitly containing '_seg' first
    seg_explicit = [c for c in seg_candidates if '_seg' in os.path.basename(c).lower()]
    if seg_explicit:
        seg_path = seg_explicit[0]
    elif seg_candidates:
        seg_path = seg_candidates[0]
    else:
        # if no seg file found, but mask exists, we'll use mask as seg_source (fallback)
        seg_path = None

    return image_path, mask_path, seg_path


def split_samples(sample_folders, train_frac=0.75, val_frac=0.05, test_frac=0.20, seed=42):
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    n = len(sample_folders)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = int(np.floor(n * val_frac))
    n_test = int(np.floor(n * test_frac))
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    train_list = [sample_folders[i] for i in train_idx]
    val_list = [sample_folders[i] for i in val_idx]
    test_list = [sample_folders[i] for i in test_idx]
    return train_list, val_list, test_list


def write_csv(list_of_pairs, csv_path):
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sample', 'h5_path', 'seg_path'])
        for name, paths in list_of_pairs:
            writer.writerow([name, paths[0], paths[1]])


def main(base_dir, output_dir, target_size=224, start_slice=30, end_slice=10,
         train_frac=0.75, val_frac=0.05, test_frac=0.20, seed=42):
    print("Preprocessing script started")
    print(f" base_dir = {base_dir}")
    print(f" output_dir = {output_dir}")
    print(f" splits: train={train_frac}, val={val_frac}, test={test_frac}")

    os.makedirs(output_dir, exist_ok=True)
    for sub in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    seg_base = os.path.join(output_dir, 'seg')
    for sub in ['train', 'val', 'test']:
        os.makedirs(os.path.join(seg_base, sub), exist_ok=True)

    sample_folders = sorted([os.path.join(base_dir, d) for d in os.listdir(base_dir)
                             if os.path.isdir(os.path.join(base_dir, d))])
    print(f" Found {len(sample_folders)} subfolder(s).")

    train_list, val_list, test_list = split_samples(sample_folders, train_frac, val_frac, test_frac, seed)
    print(f" Split: train={len(train_list)}, val={len(val_list)}, test={len(test_list)}")

    processed, skipped = 0, 0
    csv_data = {'train': [], 'val': [], 'test': []}

    def process_split(sample_list, split_name):
        nonlocal processed, skipped
        out_dir = os.path.join(output_dir, split_name)           # main h5 location
        seg_out_dir = os.path.join(seg_base, split_name)         # seg h5 location (separate)
        for folder in sample_list:
            print(f"\nProcessing [{split_name}] -> {folder}")
            img_path, mask_path, seg_src = find_image_mask_seg_in_folder(folder)
            if img_path is None or mask_path is None:
                print(f"  Skipping: missing image or mask (img={img_path}, mask={mask_path})")
                skipped += 1
                continue

            name = os.path.basename(folder.rstrip('/\\'))
            out_h5 = os.path.join(out_dir, f"{name}.h5")
            seg_h5 = os.path.join(seg_out_dir, f"{name}_seg.h5")

            try:
                preprocess_volume(img_path, mask_path, seg_src, out_h5, seg_h5,
                                  target_size, start_slice, end_slice)
                processed += 1
                csv_data[split_name].append((name, (out_h5, seg_h5)))
            except Exception as e:
                print(f"  ERROR: {e}")
                skipped += 1

    process_split(train_list, 'train')
    process_split(val_list, 'val')
    process_split(test_list, 'test')

    for split in ['train', 'val', 'test']:
        write_csv(csv_data[split], os.path.join(output_dir, f"{split}.csv"))

    print(f"\nDone. Processed={processed}, Skipped={skipped}, Total={len(sample_folders)}")
    print(f"CSV summaries written in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--target_size', type=int, default=224)
    parser.add_argument('--start_slice', type=int, default=30)
    parser.add_argument('--end_slice', type=int, default=10)
    parser.add_argument('--train_frac', type=float, default=0.75)
    parser.add_argument('--val_frac', type=float, default=0.05)
    parser.add_argument('--test_frac', type=float, default=0.20)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args.base_dir, args.output_dir, args.target_size,
         args.start_slice, args.end_slice,
         args.train_frac, args.val_frac, args.test_frac, args.seed)
