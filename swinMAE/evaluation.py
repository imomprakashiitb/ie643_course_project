import os, sys, time, glob, json
from pathlib import Path
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
import cv2
import argparse
from functools import partial
import torch.nn as nn
from skimage.segmentation import felzenszwalb
from skimage.measure import regionprops
from sklearn.metrics import precision_recall_curve, average_precision_score
from matplotlib.colors import LinearSegmentedColormap

# ---- CONFIG - edit these ----
DRIVE_BASE = "/content/drive/MyDrive/ie643_course_project_24M1644"  # your drive root
IMAGE_DIR = f"{DRIVE_BASE}/Swin_mae_data/test"  # folder with <sub>.h5
SEG_DIR = f"{DRIVE_BASE}/Swin_mae_data/seg/test"  # folder with <sub>_seg.h5
CHECKPOINT = f"{DRIVE_BASE}/swin_saved_models/checkpoint-40.pth"  # path to checkpoint
SAVE_DIR = f"{DRIVE_BASE}/swin_mae_inference"  # where to save PNGs & json
WINDOW_SIZE = 64  # sliding window size (gamma)
WINDOW_STEP = 64  # sliding step (k)
WINDOW_BATCH = 4  # how many windows to batch at once (tune for L4)
SEED = 42  # fixed seed
MIN_GT_PIXELS = 1024  # skip tiny GT slices
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# -------------------------------
print("Device:", DEVICE)
Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
# reproducible
np.random.seed(SEED)
torch.manual_seed(SEED)

# Try importing SwinMAE (user module)
try:
    from swin_mae_inference import SwinMAE
except Exception:
    if 'SwinMAE' not in globals():
        raise ImportError("SwinMAE not found. Ensure SwinMAE class is available (import or paste definition).")

# Custom yellow colormap (black -> yellow)
YELLOW_MAP = LinearSegmentedColormap.from_list("black_to_yellow", ["black", "yellow"], N=256)


# --- Utility functions ------------------------------------------------------
def coordinates_to_patch_indexes(x, y, patch_size=16, image_size=224):
    px = x // patch_size
    py = y // patch_size
    px = min(max(px, 0), image_size // patch_size - 1)
    py = min(max(py, 0), image_size // patch_size - 1)
    return px, py


def sliding_windows_for_image(image_shape, window_size, step):
    H, W = image_shape[:2]
    for y in range(48, H - window_size + 1, step):
        for x in range(25, W - window_size + 1, step):
            yield x, y


def minmax_normalization_uint8(image):
    mn = image.min(); mx = image.max()
    if mx == mn:
        return np.zeros_like(image, dtype=np.uint8)
    out = 255 * ((image - mn) / (mx - mn))
    return out.astype(np.uint8)


def reconstruction_loss_lp(orig, recon):
    if orig.ndim == 3 and orig.shape[2] == 3:
        o = orig[..., 0]
    else:
        o = orig
    if recon.ndim == 3 and recon.shape[2] == 3:
        r = recon[..., 0]
    else:
        r = recon
    return np.abs(o - r)


def compute_auprc(y_pred, y_true):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    try:
        auprc = average_precision_score(y_true.astype(int), y_pred)
        precisions, recalls, thresholds = precision_recall_curve(y_true.astype(int), y_pred)
    except Exception:
        auprc = float('nan'); precisions, recalls, thresholds = None, None, None
    return auprc, precisions, recalls, thresholds


def calculate_confusion_matrix(gt, pred):
    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    return tp, fp, fn


def calculate_dice_score(gt, pred):
    tp, fp, fn = calculate_confusion_matrix(gt, pred)
    denom = (2 * tp + fp + fn)
    dice = (2 * tp) / denom if denom > 0 else 0.0
    return dice, tp, fp, fn


# --- run_one_batch_of_windows -----------------------------------------------
def run_windows_batched(image_rgb, window_arrs, model, device):
    """
    image_rgb: np.array (H,W,3), values in float (0..1)
    window_arrs: list of lists of patch indices for each window
    Returns: list of reconstructed images (H,W,3) for each window (same order)

    This function tries to call model(x, window_arr) if supported by the loaded model class.
    If not supported, it falls back to model(x) (note: fallback DOES NOT apply window-specific masking).
    """
    recon_results = []
    x_single = torch.from_numpy(image_rgb[np.newaxis]).float().permute(0, 3, 1, 2).to(device)  # (1,3,H,W)

    for arr in window_arrs:
        with torch.no_grad():
            # prefer calling model(x, arr) if possible (the inference variant).
            try:
                loss_pred_mask = model(x_single, arr)   # expecting (loss, pred, mask)
                # if model returns a single tensor (some variants), handle below
            except TypeError:
                # fallback: model probably only accepts (x,)
                loss_pred_mask = model(x_single)

            # normalize all possible return shapes:
            # common expected: (loss, pred, mask) where pred shape (N, L, p*p*3)
            if isinstance(loss_pred_mask, tuple) or isinstance(loss_pred_mask, list):
                _, pred, _ = loss_pred_mask
            else:
                # model returned only pred (or only output); assume it's pred
                pred = loss_pred_mask

            # reconstruct image
            recon_t = model.unpatchify(pred)                 # (1,3,H,W) or similar
            recon_np = recon_t.detach().cpu().numpy()[0]     # (3,H,W)
            recon_hw3 = np.transpose(recon_np, (1, 2, 0))
            recon_results.append(recon_hw3)

            # free
            del pred, recon_t, recon_np
            torch.cuda.empty_cache()

    return recon_results
# --- main evaluation per-slice using sliding windows -------------------------
def evaluate_slice_with_sliding_window(image_rgb, label_mask, model,
                                       window_size=WINDOW_SIZE, step=WINDOW_STEP,
                                       window_batch=WINDOW_BATCH, device=DEVICE,
                                       save_dir=None, base_name="subj", slice_idx=0):
    """
    Compute combined heatmap for a single slice using sliding windows.
    Returns: combined_heatmap (H, W) float32 in [0..1].
    Also returns a 'pseudo reconstruction' (see note).
    """
    H, W = image_rgb.shape[:2]
    img_float = image_rgb.astype(np.float32)
    if img_float.max() > 1.0:
        img_float = img_float / 255.0

    sum_heat = np.zeros((H, W), dtype=np.float32)
    coverage = np.zeros((H, W), dtype=np.float32)

    windows = [(x, y) for (x, y) in sliding_windows_for_image((H, W), window_size, step)]
    window_patch_idx_list = []
    for (x, y) in windows:
        patch_idxs = []
        for a in range(x, x + window_size):
            for b in range(y, y + window_size):
                px, py = coordinates_to_patch_indexes(a, b, patch_size=16, image_size=224)
                final_patch_number = px * (224 // 16) + py
                patch_idxs.append(final_patch_number)
        window_patch_idx_list.append(list(np.unique(patch_idxs)))

    i = 0
    N = len(window_patch_idx_list)
    last_recon_full = None
    while i < N:
        batch_arrs = window_patch_idx_list[i:i + window_batch]
        batch_windows = windows[i:i + window_batch]

        recon_list = run_windows_batched(img_float, batch_arrs, model, device)

        for j, (x, y) in enumerate(batch_windows):
            recon_win = recon_list[j][y:y + window_size, x:x + window_size, :]
            orig_win = img_float[y:y + window_size, x:x + window_size, :]

            o = orig_win[..., 0].astype(np.float32)
            r = recon_win[..., 0].astype(np.float32)

            loss_map = np.abs(o - r).astype(np.float32)
            sum_heat[y:y + window_size, x:x + window_size] += loss_map
            coverage[y:y + window_size, x:x + window_size] += 1.0

        # keep last recon_list's first returned recon as a convenient "example" to visualize full-image reconstruction
        # NOTE: this is NOT the stitched full reconstruction; it's just the model output for the last processed window call's input image.
        # For a true full-image reconstruction you should run model(x_full, window_arr=None or specific) and unpatchify its pred.
        try:
            last_recon_full = recon_list[0]  # shape (H,W,3)
        except Exception:
            last_recon_full = None

        del recon_list, batch_arrs
        torch.cuda.empty_cache()
        i += window_batch

    coverage_safe = np.where(coverage == 0, 1.0, coverage)
    avg_heat = sum_heat / coverage_safe
    avg_heat = avg_heat * (img_float[..., 0] > 0.01)

    nonzero_vals = avg_heat[avg_heat > 0]
    if nonzero_vals.size > 0:
        p = np.percentile(nonzero_vals, 99.5)
        if p <= 0:
            p = nonzero_vals.max() if nonzero_vals.max() > 0 else 1.0
    else:
        p = 1.0

    combined_heatmap = np.clip(avg_heat / p, 0.0, 1.0).astype(np.float32)

    # build a pseudo-reconstruction for visualization only (original attenuated by heatmap)
    # (This is a stand-in when you don't have a stitched real reconstruction.)
    pseudo_recon = img_float.copy()
    pseudo_recon[..., 0] = np.clip(img_float[..., 0] * (1.0 - combined_heatmap), 0.0, 1.0)
    pseudo_recon = np.stack([pseudo_recon[..., 0]] * 3, axis=-1)

    # prefer last_recon_full (model output) if available (not stitched), fallback to pseudo_recon
    recon_for_vis = last_recon_full if (last_recon_full is not None) else pseudo_recon

    return combined_heatmap, recon_for_vis


# --- post-processing & saving visualizations --------------------------------
def post_process_and_save(comb_heat_map, org_img, gt, recon_img,
                          save_dir, subj_name, slice_num,
                          heatmap_dpi=140, heatmap_figsize=(12, 4),
                          compact_dpi=120, compact_figsize=(12, 3)):
    """
    Save:
     - <subj>_slice_<NNN>_anomaly.png        : Atropos seg | Combined heatmap (yellow) | Final anomaly segmentation
    And return (gt_u8, anomaly_mask).

    NOTE: The earlier 'reconstruction' PNG is intentionally NOT saved (to save space).
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    base = f"{subj_name}_slice_{slice_num:03d}"

    # prepare arrays
    org_gray = org_img[..., 0] if org_img.ndim == 3 else org_img
    org_gray = org_gray.astype(np.float32)
    comb_heat_map = comb_heat_map.astype(np.float32)
    gt_u8 = (gt > 0).astype(np.uint8)

    # --- Compute Atropos / segmentation or fallback ---
    try:
        import ants
        ants_image = ants.from_numpy(org_gray)
        img_ = ants.resample_image(ants_image, (224, 224), 1, 0)
        mask = ants.get_mask(img_)
        img_seg = ants.atropos(a=img_, m='[0.2,1x1]', c='[2,0]', i='kmeans[4]', x=mask)
        img_seg = img_seg['segmentation'].numpy()
        # resize back if necessary
        if img_seg.shape != org_gray.shape:
            import skimage.transform as sktf
            img_seg = sktf.resize(img_seg, org_gray.shape, order=0, preserve_range=True).astype(np.int32)
    except Exception:
        # fallback to felzenszwalb segmentation
        img_seg = felzenszwalb(org_gray, scale=75, sigma=0.1, min_size=10)

    # --- Final anomaly segmentation logic (same as your pipeline) ---
    heat_map_rev = (1 - comb_heat_map) * (org_gray > 0.3)
    kernel = np.ones((1, 1), np.uint8)
    eroded_image = cv2.morphologyEx((heat_map_rev * 255).astype('uint8'), cv2.MORPH_ERODE, kernel)
    eroded_image = (eroded_image / 255) > 0.5
    segments_old = felzenszwalb(comb_heat_map, scale=75, sigma=0.8, min_size=100)
    segments = eroded_image * segments_old
    region_props = regionprops(segments, intensity_image=comb_heat_map)
    intensity_sorted_regions = sorted(region_props, key=lambda prop: prop.intensity_mean, reverse=True)
    top_regions = intensity_sorted_regions[:7]
    predicted_mask_comb = np.zeros_like(gt_u8)
    for rr in top_regions:
        predicted_mask_comb = predicted_mask_comb + (segments == rr.label)
    kernel = np.ones((5, 5), np.uint8)
    predicted_mask_comb = cv2.morphologyEx(predicted_mask_comb.astype('uint8'), cv2.MORPH_DILATE, kernel)

    anomaly_mask = (predicted_mask_comb > 0).astype(np.uint8)

    # --- PNG: Anomaly visualization only (reduced size) ---
    fig, axes = plt.subplots(1, 3, figsize=heatmap_figsize, dpi=heatmap_dpi)
    # Atropos / segmentation
    axes[0].imshow(org_gray, cmap='gray')
    axes[0].imshow(img_seg, alpha=0.5, cmap='nipy_spectral')
    axes[0].set_title("Atropos / segmentation"); axes[0].axis('off')
    # Combined heatmap with yellow colormap
    axes[1].imshow(org_gray, cmap='gray')
    axes[1].imshow(comb_heat_map, cmap=YELLOW_MAP, alpha=0.7)
    axes[1].set_title("Combined heatmap (yellow)"); axes[1].axis('off')
    # Final anomaly segmentation overlay
    axes[2].imshow(org_gray, cmap='gray')
    axes[2].imshow(anomaly_mask, alpha=0.5, cmap='hot')
    axes[2].set_title("Final anomaly segmentation"); axes[2].axis('off')

    anomaly_png = os.path.join(save_dir, f"{base}_anomaly.png")
    plt.tight_layout(pad=0)
    fig.savefig(anomaly_png, bbox_inches='tight', pad_inches=0)
    plt.close(fig)

    return gt_u8, anomaly_mask


# --- run over dataset -------------------------------------------------------
def run_evaluation(image_dir, seg_dir, checkpoint, save_dir,
                   window_size=WINDOW_SIZE, step=WINDOW_STEP, window_batch=WINDOW_BATCH,
                   device=DEVICE, min_gt_pixels=MIN_GT_PIXELS, n_to_process=None):
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Instantiate & load model as before
    model = SwinMAE(norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=True)
    torch.serialization.add_safe_globals([argparse.Namespace])
    ck = torch.load(checkpoint, map_location='cpu')
    if isinstance(ck, dict) and 'model' in ck:
        st = ck['model']
    elif isinstance(ck, dict) and 'state_dict' in ck:
        st = ck['state_dict']
    else:
        st = ck
    new_st = {}
    for k, v in st.items():
        nk = k[len('module.'):] if k.startswith('module.') else k
        new_st[nk] = v
    model.load_state_dict(new_st, strict=False)
    model.to(device)
    model.eval()

    subj_files = sorted(glob.glob(os.path.join(image_dir, "*.h5")))
    subj_names = [os.path.basename(p).replace('.h5', '') for p in subj_files]
    if n_to_process is not None:
        subj_names = subj_names[:n_to_process]
    print("Subjects to evaluate:", len(subj_names))

    # --- Load existing results (if any) so we can resume ---
    results_path = os.path.join(save_dir, "eval_results.json")
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as jf:
                results = json.load(jf)
            # derive processed subjects set
            processed_subjects = set([v["subject"] + f"__slice_{v['slice']}" for v in results.values()])
            # ensure numeric keys remain consistent; set count to max key + 1
            existing_keys = [int(k) for k in results.keys()] if len(results) > 0 else []
            count = max(existing_keys) + 1 if existing_keys else 0
            print(f"Loaded existing results with {len(results)} entries, resuming at count={count}.")
        except Exception as e:
            print("Failed to load existing results json, starting fresh. Error:", e)
            results = {}
            processed_subjects = set()
            count = 0
    else:
        results = {}
        processed_subjects = set()
        count = 0

    for subj in subj_names:
        print(f"\n>>> Subject: {subj}")
        image_path = os.path.join(image_dir, f"{subj}.h5")
        seg_path = os.path.join(seg_dir, f"{subj}_seg.h5")
        if not os.path.exists(image_path) or not os.path.exists(seg_path):
            print(" Missing files, skipping:", image_path, seg_path)
            continue
        with h5py.File(image_path, 'r') as f_img, h5py.File(seg_path, 'r') as f_seg:
            img_ds = f_img['image'] # shape (1,H,W,D)
            seg_key = 'label' if 'label' in f_seg else ('prediction' if 'prediction' in f_seg else None)
            if seg_key is None:
                print(" seg h5 missing 'label' or 'prediction'; skipping")
                continue
            seg_ds = f_seg[seg_key]
            _, H, W, D = img_ds.shape
            for s in range(D):
                # skip if this subject+slice already processed
                subj_slice_id = subj + f"__slice_{s}"
                if subj_slice_id in processed_subjects:
                    print(f"  skipping {subj} slice {s} (already processed)")
                    continue

                orig = np.array(img_ds[0, :, :, s], dtype=np.float32)
                gt_mask = np.array(seg_ds[0, :, :, s], dtype=np.uint8)
                if gt_mask.sum() <= min_gt_pixels:
                    continue

                img_rgb = np.stack([orig, orig, orig], axis=-1)
                if img_rgb.max() > 1.0:
                    img_rgb = img_rgb / 255.0

                t0 = time.time()
                combined_heatmap, recon_for_vis = evaluate_slice_with_sliding_window(
                    img_rgb, gt_mask, model,
                    window_size=window_size, step=step, window_batch=window_batch, device=device,
                    save_dir=save_dir, base_name=subj, slice_idx=s
                )
                t1 = time.time()

                gt_out, pred_mask = post_process_and_save(
                    comb_heat_map=combined_heatmap,
                    org_img=img_rgb,
                    gt=gt_mask,
                    recon_img=recon_for_vis,
                    save_dir=save_dir,
                    subj_name=subj,
                    slice_num=s
                )

                dice, tp, fp, fn = calculate_dice_score(gt_out, pred_mask)
                auprc, precisions, recalls, _ = compute_auprc(combined_heatmap, gt_out)

                vispath = os.path.join(save_dir, f"{subj}_slice_{s:03d}_compact.png")
                # save smaller compact (the code that writes the compact image)
                fig, axes = plt.subplots(1, 4, figsize=(12, 3), dpi=120)
                axes[0].imshow(orig, cmap='gray'); axes[0].axis('off')
                axes[1].imshow(orig, cmap='gray'); axes[1].imshow(gt_out, alpha=0.35, cmap='Reds'); axes[1].axis('off')
                try:
                    axes[2].imshow(recon_for_vis[..., 0], cmap='gray')
                except Exception:
                    axes[2].imshow(orig, cmap='gray')
                axes[2].axis('off')
                axes[3].imshow(combined_heatmap, cmap=YELLOW_MAP); axes[3].axis('off')
                plt.tight_layout(pad=0); fig.savefig(vispath, bbox_inches='tight', pad_inches=0); plt.close(fig)

                # store result and flush to disk immediately (so resume point is saved)
                results[count] = {
                    "subject": subj,
                    "slice": int(s),
                    "dice": float(dice),
                    "tp": int(tp),
                    "fp": int(fp),
                    "fn": int(fn),
                    "auprc": float(auprc) if np.isfinite(auprc) else None,
                    "time_sec": float(t1 - t0),
                    "vis": vispath
                }
                # mark processed
                processed_subjects.add(subj_slice_id)

                # write out results file immediately
                with open(results_path, "w") as jf:
                    json.dump(results, jf)
                    jf.flush()

                count += 1
                # cleanup
                del combined_heatmap, pred_mask, gt_out, recon_for_vis
                torch.cuda.empty_cache()

    print("Done. Saved results to", results_path)
    return results

# ----------------- Run evaluation (entry) -----------------
if __name__ == "__main__":
    _results = run_evaluation(IMAGE_DIR, SEG_DIR, CHECKPOINT, SAVE_DIR,
                              window_size=WINDOW_SIZE, step=WINDOW_STEP, window_batch=WINDOW_BATCH,
                              device=DEVICE, min_gt_pixels=MIN_GT_PIXELS, n_to_process=None)
    print("Done. Summarized processed slices:", len(_results))
