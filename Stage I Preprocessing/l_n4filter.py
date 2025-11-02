"""
l_n4filter.py

Apply N4 bias field correction to a directory of NIfTI volumes.
Supports optional mask directory and safe in-place processing via a temporary directory.

Usage:
  # Write outputs to ./out
  python l_n4filter.py -i /path/to/imgs -o ./out -m /path/to/masks

  # Process in-place (safe): use no -o (temp dir used, final files replace originals)
  python l_n4filter.py -i /path/to/imgs -m /path/to/masks
"""
import ants
import sys
import argparse
from pathlib import Path
import tempfile
import shutil
from tqdm import tqdm

def arg_parser():
    parser = argparse.ArgumentParser(description='Run N4 bias field correction on NIfTI images')
    parser.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed')
    parser.add_argument('-o', '--out-dir', type=str, required=False, default=None,
                        help='output directory for preprocessed files (if omitted, process in-place)')
    parser.add_argument('-m', '--mask-dir', type=str, required=False, default=None,
                        help='mask directory (optional) - mask filenames should align with image names or be discoverable by simple heuristics')
    parser.add_argument('--skip-existing', action='store_true',
                        help='if set, skip processing files that already exist in out-dir')
    parser.add_argument('--n4-iters', type=int, nargs='+', default=[200,200,200,200],
                        help='N4 iteration schedule (default: 200 200 200 200)')
    parser.add_argument('--n4-tol', type=float, default=5e-4,
                        help='N4 tolerance (default 0.0005)')
    parser.add_argument('--smooth-sigma', type=float, default=1.0,
                        help='sigma for smoothing mask (default 1.0)')
    return parser

def strip_nii_extensions(name: str) -> str:
    if name.endswith('.nii.gz'):
        return name[:-7]
    if name.endswith('.nii'):
        return name[:-4]
    return name

def find_mask_for_file(img_name: str, mask_dir: Path):
    """
    Try to find a matching mask file in mask_dir for a given image filename.
    Heuristics:
      - Replace common modality suffixes (_t1,_t2,_flair,_dwi,_t1ce) with _mask
      - Try stem + '_mask' with .nii/.nii.gz
      - Fallback: glob any file containing the image stem and 'mask'
    Returns Path or None.
    """
    if mask_dir is None:
        return None
    stem = strip_nii_extensions(img_name)
    # Known modality suffixes to replace
    candidates_bases = [stem + '_mask']
    # If stem ends with a modality like _t1, _t2 etc, also try replacements
    for mod in ['_t1ce', '_t2', '_t1', '_flair', '_FLAIR', '_dwi']:
        if stem.endswith(mod):
            base_alt = stem[:-len(mod)] + '_mask'
            candidates_bases.append(base_alt)
    # Build candidates
    candidates = []
    for base in candidates_bases:
        candidates.append(mask_dir / (base + '.nii'))
        candidates.append(mask_dir / (base + '.nii.gz'))
        candidates.append(mask_dir / base)  # in case user has weird naming
    for c in candidates:
        if c.exists():
            return c
    # fallback: any file containing stem and mask in name
    for c in mask_dir.glob(f'*{stem}*mask*.nii*'):
        if c.exists():
            return c
    return None

def run_n4_on_image(img_path: Path, out_path: Path, mask_path: Path = None, n4_opts=None, smooth_sigma=1.0):
    """
    Read image, optionally read & smooth mask, run N4, and write out.
    Returns True on success, False on failure.
    """
    try:
        img = ants.image_read(str(img_path))
    except Exception as e:
        print(f"[ERROR] Failed to read image {img_path}: {e}")
        return False

    weight_mask = None
    if mask_path is not None and mask_path.exists():
        try:
            mask = ants.image_read(str(mask_path))
            weight_mask = ants.smooth_image(mask, sigma=smooth_sigma)
        except Exception as e:
            print(f"[WARN] Could not read/smooth mask {mask_path}: {e}")
            weight_mask = None

    try:
        # n4_opts is a dict with 'iters' and 'tol' keys
        # Signature in ANTsPy may vary; this usage works with many versions:
        # ants.n4_bias_field_correction(img, mask=None, convergence={'iters':..., 'tol':...})
        if n4_opts is None:
            n4_opts = {'iters': [200,200,200,200], 'tol': 5e-4}
        corrected = ants.n4_bias_field_correction(img,
                                                 weight_mask=weight_mask,
                                                 convergence=n4_opts)
    except TypeError:
        # some ANTsPy versions expect different param names; fallback to basic call
        try:
            if weight_mask is not None:
                corrected = ants.n4_bias_field_correction(img, mask=weight_mask)
            else:
                corrected = ants.n4_bias_field_correction(img)
        except Exception as e:
            print(f"[ERROR] N4 failed for {img_path}: {e}")
            return False
    except Exception as e:
        print(f"[ERROR] N4 failed for {img_path}: {e}")
        return False

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ants.image_write(corrected, str(out_path))
        return True
    except Exception as e:
        print(f"[ERROR] Failed to write output {out_path}: {e}")
        return False

def main(argv=None):
    args = arg_parser().parse_args(argv)
    img_dir = Path(args.img_dir)
    out_dir = Path(args.out_dir) if args.out_dir else None
    mask_dir = Path(args.mask_dir) if args.mask_dir else None

    if not img_dir.exists() or not img_dir.is_dir():
        print(f"[ERROR] img-dir '{img_dir}' does not exist or is not a directory.")
        return 1

    n4_opts = {'iters': args.n4_iters, 'tol': args.n4_tol}

    # collect nifti files
    files = sorted([p for p in img_dir.iterdir() if p.is_file() and (p.name.endswith('.nii') or p.name.endswith('.nii.gz'))])
    if not files:
        print(f"[INFO] No NIfTI files found in {img_dir}.")
        return 0

    # If out_dir provided, write outputs there. Otherwise run in-place using a temp dir.
    in_place = out_dir is None
    tmpdir = None
    if in_place:
        tmpdir = Path(tempfile.mkdtemp(prefix="n4_tmp_"))
        target_root = tmpdir
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        target_root = out_dir

    try:
        for p in tqdm(files, desc="N4 bias-correction"):
            out_path = target_root / p.name
            if args.skip_existing and out_path.exists():
                tqdm.write(f"[SKIP] Output exists: {out_path}")
                continue

            mask_path = None
            if mask_dir is not None:
                mask_path = find_mask_for_file(p.name, mask_dir)

            success = run_n4_on_image(p, out_path, mask_path=mask_path, n4_opts=n4_opts, smooth_sigma=args.smooth_sigma)
            if not success:
                tqdm.write(f"[WARN] Failed to process {p.name}; continuing.")
                continue

            # if writing to a separate out_dir we leave files there.
            # if in_place, move processed file back to replace original
            if in_place:
                try:
                    # attempt to atomically replace original (shutil.move will overwrite on most platforms)
                    shutil.move(str(out_path), str(p))
                except Exception as e:
                    tqdm.write(f"[ERROR] Could not move {out_path} -> {p}: {e}")
    finally:
        # cleanup tempdir if used
        if tmpdir is not None:
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass

    print("Done.")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))