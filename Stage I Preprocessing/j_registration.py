import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
from tqdm import tqdm 
import argparse
import os
import sys
from pathlib import Path
import ants
import traceback

def arg_parser():
    parser = argparse.ArgumentParser(
        description='Resample / register 3D volumes in NIfTI format to a template')
    parser.add_argument('-i', '--img-dir', type=str, required=True,
                        help='path to directory with images to be processed (moving images)')
    parser.add_argument('-modal', '--modality', type=str, required=False, default='_t1',
                        help='modality suffix used in filenames (e.g. _t2, _t1, _flair)')
    parser.add_argument('-o', '--out-dir', type=str, required=False, default='tmp',
                        help='output directory for registered files')
    parser.add_argument('-r', '--resolution', type=float, required=False, nargs=3, default=[1.0, 1.0, 1.0],
                        help='target resolution (unused here, kept for compatibility)')
    parser.add_argument('-or', '--orientation', type=str, required=False, default='RAS',
                        help='target orientation (unused here)')
    parser.add_argument('-inter', '--interpolation', type=int, required=False, default=4,
                        help='interpolation (unused here)')
    parser.add_argument('-nomask', '--nomaskandseg', type=int, required=False, default=0,
                        help='set to one if you want to ignore masks/segmentations in parent folder')
    parser.add_argument('-trans', '--transform', type=str, required=False, default='Rigid',
                        help='specify the transformation (Rigid, Affine, SyN...)')
    parser.add_argument('-templ', '--template', type=str, required=True,
                        help='path to template (fixed image)')
    return parser

def is_nifti_filename(name: str):
    nl = name.lower()
    return nl.endswith('.nii') or nl.endswith('.nii.gz')

def main(args=None):
    args = arg_parser().parse_args(args)
    src_basepath = args.img_dir
    dest_basepath = args.out_dir

    if not os.path.isdir(src_basepath):
        raise ValueError('(-i / --img-dir) argument needs to be a directory of NIfTI images.')

    Path(dest_basepath).mkdir(parents=True, exist_ok=True)

    # Read and reorient fixed/template image
    fixed_im = ants.image_read(args.template)
    try:
        fixed_im = fixed_im.reorient_image2('RAI')
    except Exception:
        # reorient_image2 might not be available in all ants versions; ignore if it fails
        pass

    # Prepare mask/seg source and outputs if available in parent folder
    parent = Path(src_basepath).parents[0]
    seg_path = None
    seg_out = None
    mask_path = None
    mask_out = None

    if (parent / 'seg').is_dir() and args.nomaskandseg != 1:
        seg_path = str(parent / 'seg')
        seg_out = str(Path(dest_basepath).parents[0] / 'seg')
        Path(seg_out).mkdir(parents=True, exist_ok=True)

    if (parent / 'mask').is_dir() and args.nomaskandseg != 1:
        mask_path = str(parent / 'mask')
        mask_out = str(Path(dest_basepath).parents[0] / 'mask')
        Path(mask_out).mkdir(parents=True, exist_ok=True)

    # enumerate files: only process nifti files
    files = [f for f in sorted(os.listdir(src_basepath)) if is_nifti_filename(f)]
    if not files:
        print("No NIfTI files found in", src_basepath)
        return 0

    for i, file in enumerate(tqdm(files, desc="Registering images")):
        try:
            dest_file_path = os.path.join(dest_basepath, file)
            # Build mask/seg expected filenames (only if mask/seg sources exist)
            mask_file_src = None
            seg_file_src = None
            mask_file_out = None
            seg_file_out = None

            if mask_path:
                mask_file_src = os.path.join(mask_path, file.replace(args.modality, '_mask'))
                mask_file_out = os.path.join(mask_out, file.replace(args.modality, '_mask'))

            if seg_path:
                seg_file_src = os.path.join(seg_path, file.replace(args.modality, '_seg'))
                seg_file_out = os.path.join(seg_out, file.replace(args.modality, '_seg'))

            # Determine whether we need to (re)compute this subject
            dest_exists = os.path.isfile(dest_file_path)
            mask_exists = True if (mask_file_out is None) else os.path.isfile(mask_file_out)
            seg_exists = True if (seg_file_out is None) else os.path.isfile(seg_file_out)

            if dest_exists and mask_exists and seg_exists:
                # already done for this file
                # print informative message and skip
                print(f"[skip] destination + (mask/seg if expected) already exist for {file}")
                continue

            # Read moving image
            path_img = os.path.join(src_basepath, file)
            print(f"[{i+1}/{len(files)}] Registering {path_img} -> {dest_file_path}")
            moving_im = ants.image_read(path_img)

            # Register to template
            im_tx = ants.registration(fixed=fixed_im, moving=moving_im, type_of_transform=args.transform)
            moved_im = im_tx.get('warpedmovout', None)
            if moved_im is None:
                # fallback to 'warpedmovout' key not present in some ants versions
                moved_im = im_tx.get('warpedmovout') if isinstance(im_tx, dict) else None
            if moved_im is None:
                # try common return keys
                moved_im = im_tx.get('warpedmovout') if isinstance(im_tx, dict) else None

            # If still None, try to use 'warpedmovout' regardless (some versions vary)
            if moved_im is None:
                # try other key names
                if isinstance(im_tx, dict):
                    for k in im_tx.keys():
                        if 'warp' in k.lower() or 'warped' in k.lower():
                            moved_im = im_tx[k]
                            break

            if moved_im is None:
                raise RuntimeError("ants.registration did not return warped output in known keys.")

            ants.image_write(moved_im, dest_file_path)

            # If mask exists in the source, transform + write
            if mask_file_src and os.path.isfile(mask_file_src):
                try:
                    moving_mask = ants.image_read(mask_file_src)
                    moved_mask = ants.apply_transforms(fixed=fixed_im, moving=moving_mask,
                                                       transformlist=im_tx.get('fwdtransforms', []))
                    ants.image_write(moved_mask, mask_file_out)
                except Exception as e:
                    print("  Warning: failed to transform/write mask for", file, "-", e)

            # If seg exists in the source, transform + write
            if seg_file_src and os.path.isfile(seg_file_src):
                try:
                    moving_seg = ants.image_read(seg_file_src)
                    try:
                        moving_seg = moving_seg.reorient_image2('RAI')
                    except Exception:
                        pass
                    moved_seg = ants.apply_transforms(fixed=fixed_im, moving=moving_seg,
                                                      transformlist=im_tx.get('fwdtransforms', []))
                    ants.image_write(moved_seg, seg_file_out)
                except Exception as e:
                    print("  Warning: failed to transform/write seg for", file, "-", e)

        except Exception as e:
            print(f"ERROR processing file {file}: {e}")
            traceback.print_exc()
            # continue to next subject instead of aborting batch
            continue

    print("Registration step finished.")
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))