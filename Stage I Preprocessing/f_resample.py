#!/usr/bin/env python3
import ants
import nibabel as nib
import argparse
import os
import sys
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
import numpy as np

def arg_parser():
    parser = argparse.ArgumentParser(description='Resample 3D volumes in NIfTI format')
    parser.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed')
    parser.add_argument('-o', '--out-dir', type=str, required=False, default=None,
                        help='output directory for preprocessed files; if omitted, replace in-place using a temporary directory')
    parser.add_argument('-r', '--resolution', type=float, required=False, nargs=3, default=[1.0, 1.0, 1.0],
                        help='target resolution (e.g. -r 1.0 1.0 1.0)')
    parser.add_argument('-or', '--orientation', type=str, required=False, default=None,
                        help='target orientation (e.g. RAI). If omitted, orientation is not changed.')
    parser.add_argument('-inter', '--interpolation', type=int, required=False, default=4,
                        help='interpolation type for ants.resample_image (0=nearest, 1=linear, ...). Default=4')
    return parser

def process_file(src_path: Path, out_path: Path, target_res, orientation, interp):
    """
    Read src_path (file), resample and optionally reorient, then write to out_path.
    Returns True if success else False.
    """
    try:
        img = ants.image_read(str(src_path))
    except Exception as e:
        print(f"[ERROR] Failed to read {src_path}: {e}")
        return False

    # Compare spacing robustly
    try:
        current_spacing = tuple(img.spacing)
    except Exception:
        # fallback if spacing attribute naming differs
        try:
            current_spacing = tuple(img.get_spacing())
        except Exception:
            current_spacing = None

    if current_spacing is None or not np.allclose(current_spacing, tuple(target_res), atol=1e-6):
        try:
            # use_voxels=False -> spacing argument interpreted as spacing in mm
            img = ants.resample_image(img, tuple(target_res), use_voxels=False, interp_type=int(interp))
        except Exception as e:
            print(f"[WARNING] resample_image failed for {src_path}: {e}. Continuing with original image.")

    # Reorient (guarded — different ANTsPy versions may have different functions)
    if orientation:
        try:
            # preferred method if available
            if hasattr(img, "reorient_image2"):
                img = img.reorient_image2(str(orientation))
            else:
                # fallback to ants.reorient_image if available
                try:
                    img = ants.reorient_image(img, str(orientation))
                except Exception:
                    # If neither available, skip reorientation but warn
                    print(f"[WARNING] Reorientation to {orientation} not applied (function not available).")
        except Exception as e:
            print(f"[WARNING] Reorientation failed for {src_path}: {e}")

    # Write output
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(img, str(out_path))
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write {out_path}: {e}")
        return False

def main(argv=None):
    args = arg_parser().parse_args(argv)
    img_dir = Path(args.img_dir)
    if not img_dir.is_dir():
        print("(-i / --img-dir) argument needs to be a directory of NIfTI images.")
        return 1

    target_res = tuple(args.resolution)
    orientation = args.orientation
    interp = args.interpolation

    # Build list of candidate files
    files = sorted([p for p in img_dir.iterdir() if p.is_file() and (p.name.endswith('.nii') or p.name.endswith('.nii.gz'))])
    if not files:
        print(f"No NIfTI files found in {img_dir}")
        return 0

    # If out_dir provided, use it. Otherwise process in-place via a secure temp dir and then move back.
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for p in tqdm(files, desc="Resampling files"):
            dest = out_dir / p.name
            # skip if already present
            if dest.exists():
                print(f"[INFO] Skipping existing output {dest}")
                continue
            success = process_file(p, dest, target_res, orientation, interp)
            if not success:
                print(f"[WARNING] Failed to process {p}.")
    else:
        # in-place mode: use a temporary directory to write outputs then move them back atomically
        tmpdir = Path(tempfile.mkdtemp(prefix="resample_tmp_"))
        try:
            for p in tqdm(files, desc="Resampling files (in-place)"):
                tmp_out = tmpdir / p.name
                success = process_file(p, tmp_out, target_res, orientation, interp)
                if not success:
                    print(f"[WARNING] Failed to process {p}. Skipping moving this file back.")
                    continue
                # move the tmp file back replacing original
                try:
                    # remove original first (or rename, but remove ensures replace)
                    p.unlink()  # remove original
                except Exception:
                    # if unable to unlink, try to overwrite by moving (may fail across volumes)
                    pass
                try:
                    shutil.move(str(tmp_out), str(p))
                except Exception as e:
                    print(f"[ERROR] Failed to move {tmp_out} -> {p}: {e}")
            # cleanup tmp dir
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
