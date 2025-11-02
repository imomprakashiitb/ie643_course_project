#!/usr/bin/env python3
"""
g_create_zero_seg_masks.py

Traverse an input root folder (IXI) and for every *_t2.nii or *_t2.nii.gz
create a corresponding zero-valued segmentation mask named <subject>_seg.nii.gz
in the same folder.

Usage:
    python g_create_zero_seg_masks.py <input_root> [--overwrite]

Example:
    python g_create_zero_seg_masks.py "E:\Study\...\IXI" --overwrite
"""
import argparse
from pathlib import Path
import SimpleITK as sitk
import sys

def find_t2_files(root: Path):
    for p in root.rglob("*"):
        if p.is_file():
            name = p.name.lower()
            if name.endswith("_t2.nii") or name.endswith("_t2.nii.gz"):
                yield p

def make_zero_mask_for_t2(t2_path: Path, out_path: Path, overwrite: bool = False):
    if out_path.exists() and not overwrite:
        print(f"SKIP (exists): {out_path}")
        return False
    try:
        img = sitk.ReadImage(str(t2_path))
    except Exception as e:
        print(f"ERROR reading {t2_path}: {e}")
        return False

    size = img.GetSize()            # (x, y, z)
    spacing = img.GetSpacing()
    origin = img.GetOrigin()
    direction = img.GetDirection()

    # create unsigned 8-bit image with same size and metadata
    mask = sitk.Image(size, sitk.sitkUInt8)
    mask.SetSpacing(spacing)
    mask.SetOrigin(origin)
    mask.SetDirection(direction)

    # All zeros by default; if you prefer explicit fill:
    # mask = sitk.Image(size, sitk.sitkUInt8)
    # mask = sitk.Paste(mask, sitk.Image(size, sitk.sitkUInt8) * 0, ...)
    try:
        # SimpleITK auto-compresses when filename ends with .gz
        sitk.WriteImage(mask, str(out_path))
        print(f"WROTE mask: {out_path}")
        return True
    except Exception as e:
        print(f"ERROR writing {out_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Create zero seg masks for IXI T2 files")
    parser.add_argument("input_root", help="Path to IXI root folder")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks")
    args = parser.parse_args()

    root = Path(args.input_root).expanduser().resolve()
    if not root.exists():
        print("Input root not found:", root)
        sys.exit(1)

    t2_files = list(find_t2_files(root))
    if not t2_files:
        print("No *_t2.nii[.gz] files found under", root)
        return

    print(f"Found {len(t2_files)} T2 files. Processing...")

    created = 0
    skipped = 0
    errors = 0
    for t2 in sorted(t2_files):
        # output name: replace _t2.nii(.gz) with _seg.nii.gz
        name = t2.name
        if name.lower().endswith("_t2.nii.gz"):
            out_name = name[:-10] + "_seg.nii.gz"   # remove "_t2.nii.gz"
        elif name.lower().endswith("_t2.nii"):
            out_name = name[:-7] + "_seg.nii.gz"    # remove "_t2.nii"
        else:
            # should not happen due to finder
            out_name = name + "_seg.nii.gz"

        out_path = t2.parent / out_name
        ok = make_zero_mask_for_t2(t2, out_path, overwrite=args.overwrite)
        if ok:
            created += 1
        else:
            # determine if skipped or error by existence
            if out_path.exists() and not args.overwrite:
                skipped += 1
            else:
                errors += 1

    print(f"\nDone. created={created}, skipped={skipped}, errors={errors}")

if __name__ == "__main__":
    main()
