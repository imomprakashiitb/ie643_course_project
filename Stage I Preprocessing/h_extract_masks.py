import sys
import argparse
import os
from pathlib import Path


import argparse
import os
from pathlib import Path 
import sys

def parse_args():
    p = argparse.ArgumentParser(description="Extract or create brain masks from folder")
    p.add_argument("-i", "--img-dir", required=True, help="Path to directory with images to be processed")
    p.add_argument("-o", "--out-dir", required=True, help="Output directory for mask files")
    p.add_argument("--force-create", action="store_true", help="Always (re)create masks from bet or image even if mask exists")
    return p.parse_args()

def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def move_existing_masks(inp: Path, out: Path):
    moved = []
    for f in sorted(inp.iterdir()):
        if f.is_file() and (f.name.endswith('_mask.nii') or f.name.endswith('_mask.nii.gz')):
            dst = out / f.name
            f.rename(dst)
            moved.append(dst)
            print(f"MOVED existing mask: {f.name} -> {dst}")
    return moved

def subject_name_from_filename(fn: str):
    # remove extensions .nii.gz or .nii
    if fn.lower().endswith(".nii.gz"):
        base = fn[:-7]
    elif fn.lower().endswith(".nii"):
        base = fn[:-4]
    else:
        base = os.path.splitext(fn)[0]
    # drop trailing modality like _t2 or -T2 if present (common patterns)
    for suffix in ["_t2","-t2","_T2","-T2","_flair","_t1ce","_t1"]:
        if base.lower().endswith(suffix):
            base = base[:-len(suffix)]
    return base

def create_mask_from_image(img_path: Path, out_mask_path: Path):
    # lazy import heavy libs only when needed
    import nibabel as nb
    import numpy as np
    from scipy import ndimage

    print(f"Creating mask from image: {img_path.name} -> {out_mask_path.name}")
    img = nb.load(str(img_path))
    data = img.get_fdata()
    mask = (data != 0).astype(np.uint8)

    # morphological cleaning
    try:
        mask = ndimage.binary_fill_holes(mask).astype(np.uint8)
        struct = ndimage.generate_binary_structure(3,1)
        struct = ndimage.iterate_structure(struct, 2)
        mask = ndimage.binary_closing(mask, structure=struct).astype(np.uint8)
    except Exception as e:
        print("Warning: morphological cleaning failed:", e)

    mimg = nb.Nifti1Image(mask, img.affine)
    nb.save(mimg, str(out_mask_path))
    print(f"WROTE mask: {out_mask_path} (voxels true: {int(mask.sum())})")
    return out_mask_path

def main():
    args = parse_args()
    inp = Path(args.img_dir)
    out = Path(args.out_dir)
    if not inp.exists() or not inp.is_dir():
        print("Error: input directory does not exist or is not a directory:", inp)
        return 1
    safe_mkdir(out)

    # 1) Move any explicit mask files
    moved = move_existing_masks(inp, out)
    if moved and not args.force_create:
        print(f"Moved {len(moved)} existing mask(s). Done.")
        return 0

    # 2) For each image in input folder, ensure a mask exists in output folder.
    # Look for bet output first, then any nii / nii.gz
    files = [p for p in sorted(inp.iterdir()) if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))]
    if not files:
        print("No .nii or .nii.gz files found in input folder:", inp)
        return 0

    for f in files:
        subj = subject_name_from_filename(f.name)
        expected_mask = out / f"{subj}_mask.nii.gz"
        if expected_mask.exists() and not args.force_create:
            print(f"Mask already exists for {subj}: {expected_mask.name} (skipping)")
            continue

        # prefer bet files for mask creation
        if "bet" in f.name.lower():
            src = f
            create_mask_from_image(src, expected_mask)
            continue

        # otherwise, check if there exists a bet variant for this subject
        bet_variant = None
        for cand in files:
            if cand is f: 
                continue
            # same subject but contains 'bet'
            if cand.name.lower().startswith(subj.lower()) and 'bet' in cand.name.lower():
                bet_variant = cand
                break
        if bet_variant is not None:
            create_mask_from_image(bet_variant, expected_mask)
            continue

        # fallback: create mask from this file itself
        create_mask_from_image(f, expected_mask)

    print("All masks processed.")
    return 0

if __name__ == "__main__":
    sys.exit(main())