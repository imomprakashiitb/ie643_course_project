"""
k_cut_patched.py

Crops T2 images, masks, and optional segmentation files to the mask bounding box.
Supports optional padding to make all crops the same shape, and preserves/corrects
NIfTI affine so voxel->world mapping remains correct after cropping/padding.

Usage examples:
  # Crop and save outputs to ./out without padding
  python k_cut_patched.py -i /path/to/t2 -m /path/to/mask -o ./out

  # Crop and pad outputs to the maximum observed bbox across dataset
  python k_cut_patched.py -i /path/to/t2 -m /path/to/mask -o ./out --pad-to-max

Notes:
  - segmentation files are searched under sibling directory <mask_dir>/../seg
  - supports both .nii and .nii.gz
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import sys
from tqdm import tqdm


def bbox_from_mask(mask: np.ndarray):
    """Compute bounding box indices (min, max) along each axis for nonzero voxels.

    Returns:
        (mins, maxs) each a length-3 integer array, or None if mask empty.
    """
    nonzero = np.argwhere(mask > 0)
    if nonzero.size == 0:
        return None
    mins = nonzero.min(axis=0).astype(int)
    maxs = nonzero.max(axis=0).astype(int)
    return mins, maxs


def strip_nii_extensions(name: str) -> str:
    """Remove .nii or .nii.gz from filename string."""
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return name


def find_seg(mask_path: Path, mask_dir: Path):
    """Find matching segmentation file for given mask.

    Expects segmentation files under sibling directory 'seg' (mask_dir.parent/'seg').
    Returns Path or None.
    """
    seg_dir = mask_dir.parent / "seg"
    if not seg_dir.exists() or not seg_dir.is_dir():
        return None

    base = strip_nii_extensions(mask_path.name)
    stem = base.replace("_mask", "")

    candidates = [
        seg_dir / f"{stem}_seg.nii",
        seg_dir / f"{stem}_seg.nii.gz",
        seg_dir / f"{stem}_final_seg.nii",
        seg_dir / f"{stem}_final_seg.nii.gz",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback: any seg file containing the stem
    for c in seg_dir.glob(f"*{stem}*seg*.nii*"):
        if c.exists():
            return c
    return None


def save_nifti_with_affine(data: np.ndarray,
                          ref_img: nib.Nifti1Image,
                          out_path: Path,
                          crop_start=(0, 0, 0),
                          pad_before=(0, 0, 0)):
    """Save `data` using ref_img header but adjust affine so voxel->world coordinates are correct.

    Args:
        data: numpy array to save.
        ref_img: reference nibabel image (for affine/header/dtype reference).
        out_path: Path to write NIfTI file.
        crop_start: (imin, jmin, kmin) - starting voxel indices of the crop in the original image.
        pad_before: number of voxels added BEFORE the crop along each axis (for padding).
    """
    # preserve dtype if possible
    try:
        target_dtype = ref_img.get_data_dtype()
    except Exception:
        target_dtype = data.dtype
    save_data = data.astype(target_dtype, copy=False)

    # copy header and update shape
    hdr = ref_img.header.copy()
    hdr.set_data_shape(save_data.shape)

    # compute new affine: shift origin by R @ (crop_start - pad_before)
    R = ref_img.affine[:3, :3]
    orig_origin = ref_img.affine[:3, 3]
    crop_start = np.asarray(crop_start, dtype=float)
    pad_before = np.asarray(pad_before, dtype=float)
    shift = R.dot(crop_start - pad_before)
    new_affine = ref_img.affine.copy()
    new_affine[:3, 3] = orig_origin + shift

    # ensure parent exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_img = nib.Nifti1Image(save_data, new_affine, header=hdr)
    nib.save(out_img, str(out_path))


def find_image_for_mask(mask_path: Path, img_dir: Path, mode: str):
    """Try a few common candidate names for the image corresponding to mask."""
    base = strip_nii_extensions(mask_path.name)
    candidate_base = base.replace("_mask", f"_{mode}")

    candidates = [
        img_dir / candidate_base,
        img_dir / (candidate_base + ".nii"),
        img_dir / (candidate_base + ".nii.gz"),
        img_dir / (candidate_base + ".gz"),
    ]
    for c in candidates:
        if c.exists():
            return c
    # fallback: search for any file in img_dir containing the base stem
    for c in img_dir.glob(f"*{candidate_base}*.nii*"):
        if c.exists():
            return c
    return None


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Crop volumes to brain region using mask.")
    parser.add_argument("-i", "--img-dir", required=True, help="Path to T2 image directory")
    parser.add_argument("-m", "--mask-dir", required=True, help="Path to mask directory")
    parser.add_argument("-o", "--output", required=True, help="Output root directory")
    parser.add_argument("-mode", "--mode", default="t2", help="Modality name (default: t2)")
    parser.add_argument("--pad-to-max", action="store_true", help="Pad crops to the maximum observed bbox across dataset")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    img_dir = Path(args.img_dir)
    mask_dir = Path(args.mask_dir)
    out_root = Path(args.output)
    mode = args.mode

    # Prepare output structure
    out_img = out_root / mode
    out_mask = out_root / "mask"
    out_seg = out_root / "seg"
    for d in [out_root, out_img, out_mask, out_seg]:
        d.mkdir(parents=True, exist_ok=True)

    # Collect mask files (both .nii and .nii.gz)
    mask_files = sorted(list(mask_dir.glob("*_mask.nii*")))
    if not mask_files:
        print(f"No mask files found in {mask_dir}")
        return 1

    # If padding to max dims requested, first pass to compute max dims
    max_dims = np.array([0, 0, 0], dtype=int)
    if args.pad_to_max:
        for mask_path in tqdm(mask_files, desc="Scanning masks for max bbox"):
            try:
                mask_img = nib.load(str(mask_path))
                mask_data = mask_img.get_fdata()
            except Exception as e:
                print(f"[WARN] Could not load mask {mask_path}: {e}")
                continue
            bbox = bbox_from_mask(mask_data)
            if bbox is None:
                continue
            (imin, jmin, kmin), (imax, jmax, kmax) = bbox
            dims = np.array([imax - imin + 1, jmax - jmin + 1, kmax - kmin + 1], dtype=int)
            max_dims = np.maximum(max_dims, dims)
        if np.all(max_dims == 0):
            print("No nonzero voxels found in any mask.")
            return 1

    # Process masks and crop corresponding images / segs
    for mask_path in tqdm(mask_files, desc="Cropping"):
        try:
            mask_img = nib.load(str(mask_path))
            mask_data = mask_img.get_fdata()
        except Exception as e:
            print(f"[ERROR] Cannot load mask {mask_path}: {e}")
            continue

        bbox = bbox_from_mask(mask_data)
        if bbox is None:
            print(f"[WARN] Empty mask: {mask_path.name} -- skipping")
            continue
        (imin, jmin, kmin), (imax, jmax, kmax) = bbox

        # compute crop dims
        dims = np.array([imax - imin + 1, jmax - jmin + 1, kmax - kmin + 1], dtype=int)

        # Find corresponding image filename
        img_path = find_image_for_mask(mask_path, img_dir, mode)
        if img_path is None:
            print(f"[WARN] Image not found for mask {mask_path.name} -> expected {mask_path.stem.replace('_mask', '_' + mode)}(.nii/.nii.gz)")
        else:
            try:
                img = nib.load(str(img_path))
                img_data = img.get_fdata()
                img_crop = img_data[imin:imax+1, jmin:jmax+1, kmin:kmax+1]

                pad_before = (0, 0, 0)
                if args.pad_to_max:
                    target = tuple(max_dims.tolist())
                    cur = img_crop.shape
                    pads = []
                    pad_before_vals = []
                    for cur_s, tar_s in zip(cur, target):
                        total_pad = max(0, tar_s - cur_s)
                        before = total_pad // 2
                        after = total_pad - before
                        pads.append((before, after))
                        pad_before_vals.append(before)
                    img_crop = np.pad(img_crop, pads, mode='constant', constant_values=0)
                    pad_before = tuple(pad_before_vals)
                # save: adjust affine using crop_start and pad_before
                out_image_name = strip_nii_extensions(img_path.name) + ('.nii.gz' if img_path.name.endswith('.gz') else '.nii')
                save_nifti_with_affine(img_crop, img, out_img / out_image_name, crop_start=(imin, jmin, kmin), pad_before=pad_before)
            except Exception as e:
                print(f"[ERROR] Processing image {img_path}: {e}")

        # Crop mask and save (preserve mask dtype + affine)
        try:
            mask_crop = mask_data[imin:imax+1, jmin:jmax+1, kmin:kmax+1]
            pad_before_mask = (0, 0, 0)
            if args.pad_to_max:
                cur = mask_crop.shape
                target = tuple(max_dims.tolist())
                pads = []
                pad_before_vals = []
                for cur_s, tar_s in zip(cur, target):
                    total_pad = max(0, tar_s - cur_s)
                    before = total_pad // 2
                    after = total_pad - before
                    pads.append((before, after))
                    pad_before_vals.append(before)
                mask_crop = np.pad(mask_crop, pads, mode='constant', constant_values=0)
                pad_before_mask = tuple(pad_before_vals)
            save_nifti_with_affine(mask_crop, mask_img, out_mask / mask_path.name, crop_start=(imin, jmin, kmin), pad_before=pad_before_mask)
        except Exception as e:
            print(f"[ERROR] Saving cropped mask for {mask_path.name}: {e}")

        # Crop segmentation (if exists) and save
        seg_path = find_seg(mask_path, mask_dir)
        if seg_path:
            try:
                seg_img = nib.load(str(seg_path))
                seg_data = seg_img.get_fdata()
                seg_crop = seg_data[imin:imax+1, jmin:jmax+1, kmin:kmax+1]
                pad_before_seg = (0, 0, 0)
                if args.pad_to_max:
                    cur = seg_crop.shape
                    target = tuple(max_dims.tolist())
                    pads = []
                    pad_before_vals = []
                    for cur_s, tar_s in zip(cur, target):
                        total_pad = max(0, tar_s - cur_s)
                        before = total_pad // 2
                        after = total_pad - before
                        pads.append((before, after))
                        pad_before_vals.append(before)
                    seg_crop = np.pad(seg_crop, pads, mode='constant', constant_values=0)
                    pad_before_seg = tuple(pad_before_vals)
                save_nifti_with_affine(seg_crop, seg_img, out_seg / seg_path.name, crop_start=(imin, jmin, kmin), pad_before=pad_before_seg)
            except Exception as e:
                print(f"[WARN] Could not crop/save segmentation {seg_path.name}: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())