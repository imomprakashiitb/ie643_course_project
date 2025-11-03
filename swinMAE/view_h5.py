import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def view_h5_all_slices(h5_path):
    """View all slices of an MRI from a .h5 file with left/right navigation."""
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 file not found: {h5_path}")

    with h5py.File(h5_path, 'r') as f:
        if 'image' in f and 'label' in f:
            img = np.array(f['image'])[0]
            mask = np.array(f['label'])[0]
        elif 'prediction' in f:
            img = np.array(f['prediction'])[0]
            mask = np.zeros_like(img)  # dummy mask for overlay off
            print("⚠️  Viewing segmentation file (prediction only)")
        else:
            raise KeyError("No 'image' or 'prediction' dataset found in this .h5 file.")
        print(f"Loaded: {os.path.basename(h5_path)}")
        print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")

    num_slices = img.shape[2]
    slice_idx = 0

    fig, ax = plt.subplots(figsize=(6, 6))

    def show_slice(idx):
        ax.clear()
        ax.imshow(img[:, :, idx], cmap='gray', vmin=0, vmax=1)
        ax.imshow(np.ma.masked_where(mask[:, :, idx] == 0, mask[:, :, idx]),
                  cmap='autumn', alpha=0.5)
        ax.set_title(f"Slice {idx+1}/{num_slices}")
        ax.axis('off')
        fig.canvas.draw_idle()

    def on_key(event):
        nonlocal slice_idx
        if event.key == 'right':
            slice_idx = (slice_idx + 1) % num_slices
            show_slice(slice_idx)
        elif event.key == 'left':
            slice_idx = (slice_idx - 1) % num_slices
            show_slice(slice_idx)
        elif event.key == 'q':
            plt.close(fig)

    fig.canvas.mpl_connect('key_press_event', on_key)
    show_slice(slice_idx)

    print("\nNavigation:")
    print("  → : next slice")
    print("  ← : previous slice")
    print("  q : quit viewer\n")

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_path", required=True, help="Path to .h5 file to view")
    args = parser.parse_args()
    view_h5_all_slices(args.h5_path)

