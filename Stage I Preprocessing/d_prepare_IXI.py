"""
Batch IXI preprocessing.

Usage:
    python c_prepare_IXI.py <input_root> <output_root> [--device {cuda,cpu}]
                           [--skip-existing] [--keep-intermediate]
"""
import argparse
import sys
import subprocess
from pathlib import Path
import shutil
import traceback
import os
import fnmatch
import time

# ---------------- helpers ----------------
def run_cmd(cmd, desc=None, check=True, env=None, timeout=None):
    print("\n>> Running:", " ".join(map(str, cmd)))
    try:
        subprocess.run(cmd, check=check, env=env, timeout=timeout)
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

def copy_dir_contents(src_dir: Path, dst_dir: Path):
    if not src_dir.exists():
        raise FileNotFoundError(f"{src_dir} does not exist")
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in src_dir.iterdir():
        if item.is_dir():
            shutil.copytree(item, dst_dir / item.name, dirs_exist_ok=True)
        else:
            dst = dst_dir / item.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dst)

def find_t2_in_folder(folder: Path):
    files = sorted([p for p in folder.iterdir() if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"))])
    candidates = []
    for p in files:
        name = p.name.lower()
        if "t2" in name:
            return p
        candidates.append(p)
    for sub in sorted([d for d in folder.iterdir() if d.is_dir()]):
        for p in sorted(sub.iterdir()):
            if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz")):
                if "t2" in p.name.lower():
                    return p
        for p in sorted(sub.iterdir()):
            if p.is_file() and (p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz")):
                return p
    if candidates:
        return candidates[0]
    return None

def infer_subject_name_from_file(p: Path):
    name = p.name
    if name.lower().endswith(".nii.gz"):
        base = name[:-7]
    elif name.lower().endswith(".nii"):
        base = name[:-4]
    else:
        base = name
    for suf in ("_t2","-t2","_T2","-T2"):
        if base.endswith(suf):
            base = base[:-len(suf)]
    return base

def run_hdbet_cli_or_fallback(input_dir: Path, output_dir: Path, device: str = "cuda"):
    venv_scripts = str(Path(sys.executable).parent)
    cli_path = shutil.which("hd-bet")
    if cli_path is None:
        candidate = Path(venv_scripts) / "hd-bet.exe"
        if candidate.exists():
            cli_path = str(candidate)

    if cli_path:
        env = os.environ.copy()
        env["PATH"] = venv_scripts + os.pathsep + env.get("PATH", "")
        cmd = [cli_path, "-i", str(input_dir), "-o", str(output_dir), "-device", device]
        return run_cmd(cmd, desc="hd-bet (CLI)", check=True, env=env, timeout=300)
    else:
        print("hd-bet CLI not found in PATH. Falling back to 'python -m hd_bet' then 'python -m HD_BET'.")
        cmd_lower = [sys.executable, "-m", "hd_bet", "-i", str(input_dir), "-o", str(output_dir), "-device", device]
        if run_cmd(cmd_lower, desc="hd-bet (python -m hd_bet)", check=False, timeout=300):
            return True
        cmd_upper = [sys.executable, "-m", "HD_BET", "-i", str(input_dir), "-o", str(output_dir), "-device", device]
        return run_cmd(cmd_upper, desc="hd-bet (python -m HD_BET)", check=False, timeout=300)

def remove_subject_intermediates(output_root: Path, subj: str):
    stage_paths = [
        output_root / "v1resampled" / "IXI" / "t2",
        output_root / "v2skullstripped" / "IXI" / "t2",
        output_root / "v2skullstripped" / "IXI" / "mask",
        output_root / "v3registered_non_iso" / "IXI" / "t2",
        output_root / "v3registered_non_iso" / "IXI" / "mask",
        output_root / "v3registered_non_iso_cut" / "IXI" / "t2",
        output_root / "v3registered_non_iso_cut" / "IXI" / "mask",
        output_root / "v4correctedN4_non_iso_cut" / "IXI" / "t2",
        output_root / "v4correctedN4_non_iso_cut" / "IXI" / "mask",
    ]
    removed = []
    for sp in stage_paths:
        if not sp.exists():
            continue
        for p in list(sp.iterdir()):
            try:
                if subj in p.name:
                    if p.is_file():
                        p.unlink()
                        removed.append(p)
                    elif p.is_dir():
                        shutil.rmtree(p)
                        removed.append(p)
            except Exception:
                pass
    return removed

# ---------------- main ----------------
def main():
    parser = argparse.ArgumentParser(description="Batch IXI preprocessing (fixed HD-BET invocation)")
    parser.add_argument("input_root", help="Folder with IXI subject folders or nifti files")
    parser.add_argument("output_root", help="Where to write preprocessed outputs")
    parser.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    parser.add_argument("--skip-existing", action="store_true", help="Skip subjects that already have final outputs")
    parser.add_argument("--keep-intermediate", action="store_true", help="Keep intermediate files (don't delete)")
    args = parser.parse_args()

    INPUT = Path(args.input_root).resolve()
    OUTPUT = Path(args.output_root).resolve()
    if not INPUT.exists():
        print("Input root not found:", INPUT)
        sys.exit(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    subject_sources = []
    for p in sorted(INPUT.iterdir()):
        if p.is_dir():
            found = any((x.suffix.lower() in (".nii",) or x.name.lower().endswith(".nii.gz")) for x in p.rglob("*"))
            if found:
                subject_sources.append(p)
        else:
            if p.name.lower().endswith(".nii") or p.name.lower().endswith(".nii.gz"):
                subject_sources.append(p)

    if not subject_sources:
        print("No subject folders or nifti files found under:", INPUT)
        sys.exit(1)

    print(f"Found {len(subject_sources)} subject sources under {INPUT}")

    for idx, src in enumerate(subject_sources, start=1):
        print("\n" + "="*80)
        print(f"[{idx}/{len(subject_sources)}] Processing source: {src}")
        try:
            if src.is_file():
                t2_file = src
            else:
                t2_file = find_t2_in_folder(src)
            if t2_file is None:
                print("  -> No t2 file found for source, skipping.")
                continue
            subj = infer_subject_name_from_file(t2_file)
            print("  -> Subject:", subj, "t2 file:", t2_file.name)

            V1_DIR = OUTPUT / "v1resampled" / "IXI" / "t2"
            V2_DIR = OUTPUT / "v2skullstripped" / "IXI" / "t2"
            V2_MASK = OUTPUT / "v2skullstripped" / "IXI" / "mask"
            V3_DIR = OUTPUT / "v3registered_non_iso" / "IXI" / "t2"
            V3_MASK = OUTPUT / "v3registered_non_iso" / "IXI" / "mask"
            V3_CUT_DIR = OUTPUT / "v3registered_non_iso_cut" / "IXI"
            V4_DIR = OUTPUT / "v4correctedN4_non_iso_cut" / "IXI" / "t2"
            V4_MASK = OUTPUT / "v4correctedN4_non_iso_cut" / "IXI" / "mask"
            FINAL_SUBJ_DIR = OUTPUT / "final_v4" / "IXI" / subj

            for d in (V1_DIR, V2_DIR, V2_MASK, V3_DIR, V3_MASK, V3_CUT_DIR, V4_DIR, V4_MASK, FINAL_SUBJ_DIR):
                d.mkdir(parents=True, exist_ok=True)

            final_t2_candidates = list(FINAL_SUBJ_DIR.glob(f"{subj}*_t2.*")) + list(FINAL_SUBJ_DIR.glob(f"{subj}*t2.*"))
            if args.skip_existing and any(final_t2_candidates):
                print("  -> Final output exists and --skip-existing set. Skipping:", subj)
                continue

            dest_t2_name = f"{subj}_t2.nii.gz" if t2_file.name.lower().endswith(".nii.gz") else f"{subj}_t2.nii"
            dest_t2 = V1_DIR / dest_t2_name
            safe_copy(t2_file, dest_t2)
            print("  -> placed sample t2 at:", dest_t2)

            ok = run_cmd([sys.executable, "Preprocessing/f_resample.py", "-i", str(V1_DIR), "-o", str(V1_DIR), "-r", "1.0", "1.0", "1.0"], desc="resample")
            if not ok:
                print("  -> resample failed for", subj, " — skipping subject.")
                continue

            ok = run_hdbet_cli_or_fallback(V1_DIR, V2_DIR, device=args.device)
            if not ok:
                print("  -> hd-bet failed for", subj, " — skipping subject.")
                continue

            ok = run_cmd([sys.executable, "Preprocessing/h_extract_masks.py", "-i", str(V2_DIR), "-o", str(V2_MASK)], desc="extract_masks")
            if not ok:
                print("  -> extract_masks failed for", subj, " — attempting to continue.")

            run_cmd([sys.executable, "Preprocessing/i_replace.py", "-i", str(V2_MASK), "-s", "_t2", ""], desc="replace mask names (non-fatal)")

            templ = "Data/Preprocessed_Input/sri_atlas/templates/t2_brain.nii"
            # Pass explicit mask folder to registration by giving -i the t2 folder (V2_DIR) and out folder V3_DIR.
            ok = run_cmd([sys.executable, "Preprocessing/j_registration.py", "-i", str(V2_DIR), "-o", str(V3_DIR), "--modality=_t2", "-trans", "Affine", "-templ", templ], desc="registration")
            if not ok:
                print("  -> registration failed for", subj, " — skipping subject.")
                continue

            # Verify registration produced masks in V3_MASK; fallback: copy from V2_MASK if missing
            v3_mask_files = list(V3_MASK.glob("*"))
            if not v3_mask_files:
                print("  -> WARNING: registration did not write masks to", V3_MASK, " — attempting to copy from", V2_MASK)
                if V2_MASK.exists():
                    for p in V2_MASK.iterdir():
                        if subj in p.name:
                            safe_copy(p, V3_MASK / p.name)
                    v3_mask_files = list(V3_MASK.glob("*"))
                if not v3_mask_files:
                    print("  -> ERROR: no masks available for", subj, " — skipping subject.")
                    continue

            # Call cut with explicit mask folder (V3_MASK) and explicit t2 folder (V3_DIR)
            ok = run_cmd([sys.executable, "Preprocessing/k_cut.py", "-i", str(V3_DIR), "-m", str(V3_MASK), "-o", str(V3_CUT_DIR), "-mode", "t2"], desc="cut to brain")
            if not ok:
                print("  -> cut failed for", subj, " — skipping subject.")
                continue

            # After cut, ensure V3_CUT_DIR/t2 and V3_CUT_DIR/mask exist and are not empty
            cut_t2_folder = Path(V3_CUT_DIR) / "t2"
            cut_mask_folder = Path(V3_CUT_DIR) / "mask"
            if not cut_t2_folder.exists() or not any(cut_t2_folder.iterdir()):
                print("  -> ERROR: cut output missing t2 files at", cut_t2_folder, " — skipping subject.")
                continue
            if not cut_mask_folder.exists() or not any(cut_mask_folder.iterdir()):
                print("  -> ERROR: cut output missing mask files at", cut_mask_folder, " — skipping subject.")
                continue

            ok = run_cmd([sys.executable, "Preprocessing/l_n4filter.py", "-i", str(cut_t2_folder), "-o", str(V4_DIR), "-m", str(cut_mask_folder)], desc="n4filter")
            if not ok:
                print("  -> n4 failed for", subj, " — skipping subject.")
                continue

            moved_any = False
            for pattern in (f"{subj}_t2.nii.gz", f"{subj}_t2.nii", f"{subj}*_t2.nii.gz", f"{subj}*_t2.nii"):
                for p in V4_DIR.glob(pattern):
                    dest = FINAL_SUBJ_DIR / p.name
                    safe_move(p, dest)
                    print(f"  -> moved t2: {p.name} -> {dest}")
                    moved_any = True
                    
            moved_any = False
            for pattern in (f"{subj}_mask.nii.gz", f"{subj}_mask.nii", f"{subj}*_mask.nii.gz", f"{subj}*_mask.nii"):
                for p in V4_MASK.glob(pattern):
                    dest = FINAL_SUBJ_DIR / p.name
                    safe_move(p, dest)
                    print(f"  -> moved mask: {p.name} -> {dest}")
                    moved_any = True

            if not moved_any:
                if (V3_CUT_DIR / "mask").exists():
                    for p in (V3_CUT_DIR / "mask").iterdir():
                        if subj in p.name:
                            dest = FINAL_SUBJ_DIR / p.name
                            safe_move(p, dest)
                            print(f"  -> fallback moved {p.name} -> {dest}")
                            moved_any = True

            if not moved_any:
                print("  -> WARNING: No final files found to move for", subj)

            if not args.keep_intermediate:
                removed = remove_subject_intermediates(OUTPUT, subj)
                print(f"  -> removed {len(removed)} intermediate files for {subj}")

            print(f"[{idx}/{len(subject_sources)}] Completed subject: {subj}")

        except Exception as exc:
            print("Unexpected error processing", src)
            traceback.print_exc()
            print("Continuing to next subject...\n")
            continue

    print("\nBatch preprocessing finished.")

if __name__ == "__main__":
    main()
