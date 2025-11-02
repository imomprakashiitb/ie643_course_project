"""
Batch BraTS21 preprocessing.

Behavior:
 - Copies T2 and SEG from INPUT -> DATA_DIR/v2skullstripped/Brats21/
 - Calls the older helper scripts (g_get_masks.py, h_extract_masks.py, i_replace.py,
   j_registration.py, k_cut.py, l_n4filter.py).
 - Copies masks/seg from v3registered_non_iso_cut to v4 folder (so v4 contains t2, mask, seg)
 - Moves final t2/mask/seg into final_v4/Brats21/<subject>/ and then deletes intermediate files
   (unless --keep-intermediate is passed).
Usage:
   python c_prepare_BraTS21.py <input_root> <output_root> [--skip-existing] [--keep-intermediate]
"""
import argparse
import sys
import subprocess
from pathlib import Path
import shutil
import os
import traceback

def run_cmd(cmd, desc=None, check=True, timeout=None, env=None):
    print("\n>> Running:", " ".join(map(str, cmd)))
    try:
        subprocess.run(cmd, check=check, timeout=timeout, env=env)
        print("   -- OK:", desc or cmd[0])
        return True
    except subprocess.CalledProcessError as e:
        print("   -- FAILED:", desc or cmd[0], " returncode:", e.returncode)
        return False
    except subprocess.TimeoutExpired:
        print("   -- TIMED OUT:", desc or cmd[0])
        return False
    except Exception as e:
        print("   -- ERROR running command:", e)
        return False

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def safe_move(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(src), str(dst))
    except Exception:
        shutil.copy2(str(src), str(dst))
        try:
            src.unlink()
        except Exception:
            pass

def remove_subject_intermediates(output_root: Path, subj: str):
    """
    Remove intermediate files for subj across the stage folders (v2, v3, v3_cut, v4).
    Returns list of removed path strings (for logging).
    """
    stage_paths = [
        output_root / "v2skullstripped" / "Brats21" / "t2",
        output_root / "v2skullstripped" / "Brats21" / "seg",
        output_root / "v2skullstripped" / "Brats21" / "mask",
        output_root / "v3registered_non_iso" / "Brats21" / "t2",
        output_root / "v3registered_non_iso" / "Brats21" / "mask",
        output_root / "v3registered_non_iso_cut" / "Brats21" / "t2",
        output_root / "v3registered_non_iso_cut" / "Brats21" / "mask",
        output_root / "v4correctedN4_non_iso_cut" / "Brats21" / "t2",
        output_root / "v4correctedN4_non_iso_cut" / "Brats21" / "mask",
        output_root / "v4correctedN4_non_iso_cut" / "Brats21" / "seg",
    ]
    removed = []
    for sp in stage_paths:
        if not sp.exists():
            continue
        for p in list(sp.iterdir()):
            try:
                name = p.name
                if subj in name:
                    if p.is_file():
                        p.unlink()
                        removed.append(str(p))
                    elif p.is_dir():
                        shutil.rmtree(p)
                        removed.append(str(p))
            except Exception:
                pass
    return removed

def find_subjects_in_input(INPUT: Path):
    """
    Support two modes:
     - INPUT has 't2' and 'seg' subdirs: iterate over files in t2 (use seg with same base)
     - INPUT contains per-subject folders: treat each folder as subject and find t2/seg inside
    Returns list of tuples: (subj_name, t2_path, seg_path_or_None)
    """
    subjects = []

    # Mode A: input has t2/ and seg/ subdirs
    t2_dir = INPUT / "t2"
    seg_dir = INPUT / "seg"
    if t2_dir.exists() and t2_dir.is_dir():
        for t2_file in sorted(p for p in t2_dir.iterdir() if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))):
            base = t2_file.name
            subj = base
            # derive base key without extensions and suffixes
            if base.lower().endswith(".nii.gz"):
                key = base[:-7]
            elif base.lower().endswith(".nii"):
                key = base[:-4]
            else:
                key = base
            # look for seg file with matching key
            seg_file = None
            if seg_dir.exists():
                # try common patterns
                for candidate in (seg_dir.glob(f"{key}*"), seg_dir.glob(f"{key}_*"), seg_dir.glob(f"*{key}*")):
                    for c in candidate:
                        if c.is_file():
                            seg_file = c
                            break
                    if seg_file:
                        break
            subjects.append((key, t2_file, seg_file))
        return subjects

    # Mode B: subject folders under INPUT
    for sub in sorted(p for p in INPUT.iterdir() if p.is_dir()):
        # try to find t2 in folder
        t2 = None
        seg = None
        # look for obvious t2 names top-level then one-level down
        for p in sorted(sub.iterdir()):
            if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz")):
                if "t2" in p.name.lower():
                    t2 = p
                elif "seg" in p.name.lower():
                    seg = p
        if t2 is None:
            # search recursively one level
            for child in sorted(sub.iterdir()):
                if child.is_dir():
                    for p in sorted(child.iterdir()):
                        if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz")):
                            if "t2" in p.name.lower():
                                t2 = p
                            elif "seg" in p.name.lower():
                                seg = p
        if t2 is None:
            continue
        # infer subj key from t2 filename
        name = t2.name
        if name.lower().endswith(".nii.gz"):
            key = name[:-7]
        elif name.lower().endswith(".nii"):
            key = name[:-4]
        else:
            key = name
        for suf in ("_t2","-t2","_T2","-T2"):
            if key.endswith(suf):
                key = key[:-len(suf)]
        subjects.append((key, t2, seg))
    return subjects

def main():
    parser = argparse.ArgumentParser(description="Batch BraTS21 preprocessing (no resample, assume skull-stripped)")
    parser.add_argument("input_root", help="Folder with BraTS input (either contains t2/ and seg/ subdirs, or subject folders)")
    parser.add_argument("output_root", help="Where to write preprocessed outputs (absolute path recommended)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if final output exists")
    parser.add_argument("--keep-intermediate", action="store_true", help="Do not remove intermediate files (for debug)")
    args = parser.parse_args()

    INPUT = Path(args.input_root).resolve()
    OUTPUT = Path(args.output_root).resolve()
    if not INPUT.exists():
        print("Input root not found:", INPUT)
        sys.exit(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    subjects = find_subjects_in_input(INPUT)
    if not subjects:
        print("No subjects found under", INPUT)
        sys.exit(1)

    print(f"Found {len(subjects)} subjects to process under {INPUT}")

    # Template path (OLD-style as requested)
    templ = "Data/Preprocessed_Input/sri_atlas/templates/t2_brain.nii"

    for idx, (subj, t2_path, seg_path) in enumerate(subjects, start=1):
        print("\n" + "="*80)
        print(f"[{idx}/{len(subjects)}] Subject: {subj}")
        try:
            # Setup stage folders (matching requested structure)
            V2_T2 = OUTPUT / "v2skullstripped" / "Brats21" / "t2"
            V2_SEG = OUTPUT / "v2skullstripped" / "Brats21" / "seg"
            V2_MASK = OUTPUT / "v2skullstripped" / "Brats21" / "mask"

            V3_T2 = OUTPUT / "v3registered_non_iso" / "Brats21" / "t2"
            V3_MASK = OUTPUT / "v3registered_non_iso" / "Brats21" / "mask"

            V3_CUT = OUTPUT / "v3registered_non_iso_cut" / "Brats21"
            V3_CUT_T2 = V3_CUT / "t2"
            V3_CUT_MASK = V3_CUT / "mask"
            V3_CUT_SEG = V3_CUT / "seg"

            V4_T2 = OUTPUT / "v4correctedN4_non_iso_cut" / "Brats21" / "t2"
            V4_MASK = OUTPUT / "v4correctedN4_non_iso_cut" / "Brats21" / "mask"
            V4_SEG = OUTPUT / "v4correctedN4_non_iso_cut" / "Brats21" / "seg"

            FINAL_SUBJ = OUTPUT / "final_v4" / "Brats21" / subj

            for d in (V2_T2, V2_SEG, V2_MASK, V3_T2, V3_MASK, V3_CUT_T2, V3_CUT_MASK, V3_CUT_SEG, V4_T2, V4_MASK, V4_SEG, FINAL_SUBJ):
                d.mkdir(parents=True, exist_ok=True)

            # If skip-existing and final present, skip
            final_candidates = list(FINAL_SUBJ.glob(f"{subj}*_t2.*")) + list(FINAL_SUBJ.glob(f"{subj}*_mask.*"))
            if args.skip_existing and any(final_candidates):
                print("  -> Final output exists and --skip-existing set, skipping", subj)
                continue

            # Copy T2 and SEG into v2 folder (direct copy as requested)
            dest_t2_name = f"{subj}_t2.nii.gz" if t2_path.name.lower().endswith(".nii.gz") else f"{subj}_t2.nii"
            dest_t2 = V2_T2 / dest_t2_name
            safe_copy(t2_path, dest_t2)
            print("  -> copied t2 to", dest_t2)

            if seg_path:
                dest_seg_name = f"{subj}_seg.nii.gz" if seg_path.name.lower().endswith(".nii.gz") else f"{subj}_seg.nii"
                dest_seg = V2_SEG / dest_seg_name
                safe_copy(seg_path, dest_seg)
                print("  -> copied seg to", dest_seg)
            else:
                print("  -> NOTE: no seg file found for subject (continuing without seg)")

            # Step: get masks / extract masks / replace  (OLD helper names)
            #  - g_get_masks.py (optional depending on your pipeline)
            run_cmd([sys.executable, "Preprocessing/g_get_masks.py", "-i", str(V2_T2), "-o", str(V2_T2), "-mod", "t2"], desc="get_masks (optional)", check=False)

            ok = run_cmd([sys.executable, "Preprocessing/h_extract_masks.py", "-i", str(V2_T2), "-o", str(V2_MASK)], desc="extract_masks")
            if not ok:
                print("  -> extract_masks failed for", subj, " — skipping")
                continue

            # Clean mask names (i_replace style, old name)
            run_cmd([sys.executable, "Preprocessing/i_replace.py", "-i", str(V2_MASK), "-s", "_t2", ""], desc="replace mask names (non-fatal)", check=False)

            # Registration (old j_registration.py)
            ok = run_cmd([sys.executable, "Preprocessing/j_registration.py", "-i", str(V2_T2), "-o", str(V3_T2), "--modality=_t2", "-trans", "Affine", "-templ", templ], desc="registration")
            if not ok:
                print("  -> registration failed for", subj, " — skipping")
                continue

            # After registration, ensure masks were written into V3_MASK; if not, attempt to copy from V2_MASK
            if not any(V3_MASK.glob("*")):
                if any(V2_MASK.glob("*")):
                    for p in V2_MASK.iterdir():
                        if subj in p.name:
                            safe_copy(p, V3_MASK / p.name)
                    print("  -> copied masks from v2 to v3 as fallback")
                else:
                    print("  -> ERROR: no masks available for registration output — skipping")
                    continue

            # Cut to brain (k_cut.py): give V3_T2 and V3_MASK explicitly
            ok = run_cmd([sys.executable, "Preprocessing/k_cut.py", "-i", str(V3_T2), "-m", str(V3_MASK), "-o", str(V3_CUT), "-mode", "t2"], desc="cut")
            if not ok:
                print("  -> cut failed for", subj, " — skipping")
                continue

            # Verify cut outputs exist
            if not (V3_CUT_T2.exists() and any(V3_CUT_T2.iterdir())):
                print("  -> ERROR: cut produced no t2 output at", V3_CUT_T2, " — skipping")
                continue
            if not (V3_CUT_MASK.exists() and any(V3_CUT_MASK.iterdir())):
                print("  -> ERROR: cut produced no mask output at", V3_CUT_MASK, " — skipping")
                continue

            # N4 correction (l_n4filter.py)
            ok = run_cmd([sys.executable, "Preprocessing/l_n4filter.py", "-i", str(V3_CUT_T2), "-o", str(V4_T2), "-m", str(V3_CUT_MASK)], desc="n4filter")
            if not ok:
                print("  -> n4filter failed for", subj, " — skipping")
                continue

            # Copy masks & seg from v3registered_non_iso_cut to v4 (so v4 has t2, mask, seg)
            # masks:
            for m in V3_CUT_MASK.glob("*"):
                safe_copy(m, V4_MASK / m.name)
            # segs:
            if V3_CUT_SEG.exists():
                for s in V3_CUT_SEG.glob("*"):
                    safe_copy(s, V4_SEG / s.name)

            # Move final files into final_v4/<subj> folder
            moved_any = False
            # t2
            for pattern in (f"{subj}_t2.nii.gz", f"{subj}_t2.nii", f"{subj}*_t2.nii.gz", f"{subj}*_t2.nii"):
                for p in V4_T2.glob(pattern):
                    dest = FINAL_SUBJ / p.name
                    safe_move(p, dest)
                    print(f"  -> moved t2: {p.name} -> {dest}")
                    moved_any = True

            # mask
            for pattern in (f"{subj}_mask.nii.gz", f"{subj}_mask.nii", f"{subj}*_mask.nii.gz", f"{subj}*_mask.nii"):
                for p in V4_MASK.glob(pattern):
                    dest = FINAL_SUBJ / p.name
                    safe_move(p, dest)
                    print(f"  -> moved mask: {p.name} -> {dest}")
                    moved_any = True

            # seg
            for pattern in (f"{subj}_seg.nii.gz", f"{subj}_seg.nii", f"{subj}*_seg.nii.gz", f"{subj}*_seg.nii"):
                for p in V4_SEG.glob(pattern):
                    dest = FINAL_SUBJ / p.name
                    safe_move(p, dest)
                    print(f"  -> moved seg: {p.name} -> {dest}")
                    moved_any = True

            if not moved_any:
                print("  -> WARNING: no final files moved for", subj)

            # Remove intermediates after moving final outputs (unless keep-intermediate)
            if not args.keep_intermediate:
                removed = remove_subject_intermediates(OUTPUT, subj)
                print(f"  -> removed {len(removed)} intermediate files for {subj}")
            else:
                print("  -> keep-intermediate flag set, not deleting intermediates")

            print(f"[{idx}/{len(subjects)}] Completed subject: {subj}")

        except Exception as exc:
            print("Unexpected error processing", subj)
            traceback.print_exc()
            print("Continuing to next subject...\n")
            continue

    print("\nBatch preprocessing finished.")

if __name__ == "__main__":
    main()
