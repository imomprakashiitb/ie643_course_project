import sys
import argparse
from pathlib import Path

def arg_parser():
    parser = argparse.ArgumentParser(
        description='Replace a substring in NIfTI filenames inside a directory (supports .nii and .nii.gz)')
    required = parser.add_argument_group('Required')
    required.add_argument('-i', '--img-dir', type=str, required=True,
                          help='path to directory with images to be processed')
    required.add_argument('-s', '--string', type=str, required=True, nargs=2,
                          metavar=('OLD', 'NEW'),
                          help='two strings: the substring to replace and the replacement (e.g. -s "_old" "_new")')
    parser.add_argument('--dry-run', action='store_true',
                        help='show planned renames but do not perform them')
    parser.add_argument('--overwrite', action='store_true',
                        help='allow overwriting existing target files')
    return parser

def is_nifti(p: Path) -> bool:
    return p.suffix == '.nii' or (p.suffixes == ['.nii', '.gz'])

def main(argv=None):
    args = arg_parser().parse_args(argv)
    img_dir = Path(args.img_dir)
    old, new = args.string

    # Basic checks
    if not img_dir.exists() or not img_dir.is_dir():
        print(f"Error: '{img_dir}' is not a directory or does not exist.")
        return 1

    # Collect files
    files = sorted([p for p in img_dir.iterdir() if p.is_file() and is_nifti(p)])
    if not files:
        print(f"No NIfTI files (.nii or .nii.gz) found in {img_dir}")
        return 0

    planned = []
    for p in files:
        # Only replace in the filename (not in parent folders)
        new_name = p.name.replace(old, new)
        if new_name == p.name:
            # no change
            continue
        new_path = p.with_name(new_name)
        planned.append((p, new_path))

    if not planned:
        print("No filenames required renaming (no occurrences of the target substring).")
        return 0

    # Print planned actions
    print("Planned renames:")
    for src, dst in planned:
        print(f"  {src.name}  ->  {dst.name}")

    if args.dry_run:
        print("\nDry-run enabled: no files were changed.")
        return 0

    # Perform renames
    for src, dst in planned:
        if dst.exists() and not args.overwrite:
            print(f"[SKIP] Target exists and overwrite not enabled: {dst}")
            continue
        try:
            if dst.exists() and args.overwrite:
                dst.unlink()
            src.rename(dst)
            print(f"[OK] {src.name} -> {dst.name}")
        except Exception as e:
            print(f"[ERROR] Failed to rename {src} -> {dst}: {e}")

    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))