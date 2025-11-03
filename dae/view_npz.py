import numpy as np
import matplotlib.pyplot as plt
import os

def inspect_npz(npz_file_path, save_plots=False, output_dir=None):
    """
    Inspect and visualize the contents of an .npz file containing T2 and segmentation slices.
    
    Args:
        npz_file_path (str): Path to the .npz file (e.g., slice_0.npz).
        save_plots (bool): If True, save visualizations as PNG files instead of displaying.
        output_dir (str): Directory to save plots if save_plots is True.
    """
    # Check if file exists
    if not os.path.exists(npz_file_path):
        print(f"Error: File {npz_file_path} does not exist.")
        return
    
    # Load .npz file
    try:
        data = np.load(npz_file_path)
        print(f"\nInspecting {npz_file_path}")
        print("Keys in .npz file:", list(data.keys()))
        
        # Print shape and basic statistics for each array
        for key in data:
            array = data[key]
            print(f"\nKey: {key}")
            print(f"  Shape: {array.shape}")
            print(f"  Data type: {array.dtype}")
            print(f"  Min value: {array.min():.4f}")
            print(f"  Max value: {array.max():.4f}")
            print(f"  Mean value: {array.mean():.4f}")
            print(f"  Unique values: {np.unique(array)}")
            
            # Prepare array for visualization (remove extra dimensions)
            img = array
            while len(img.shape) > 2:
                img = img.squeeze()  # Remove dimensions of size 1
            if len(img.shape) != 2:
                print(f"  Cannot visualize {key}: Expected 2D array, got shape {img.shape}")
                continue
            
            # Visualize or save the array
            plt.figure(figsize=(8, 4))
            cmap = 'gray' if key.lower() in ['x', 't2', 'image'] else 'jet'
            plt.imshow(img, cmap=cmap)
            plt.title(f"{key} (Slice {os.path.basename(npz_file_path)})")
            plt.colorbar()
            plt.axis('off')
            
            if save_plots and output_dir:
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, f"{key}_{os.path.basename(npz_file_path)}.png")
                plt.savefig(save_path)
                plt.close()
                print(f"  Saved plot: {save_path}")
            else:
                plt.show()
            
    except Exception as e:
        print(f"Error processing {npz_file_path}: {e}")

def inspect_multiple_npz(patient_dirs, max_files_per_patient=10, save_plots=False, output_dir=None):
    """
    Inspect multiple .npz files from one or more patient directories.
    
    Args:
        patient_dirs (list): List of paths to patient directories (e.g., patient_BraTS2021_00000).
        max_files_per_patient (int): Maximum number of .npz files to process per patient.
        save_plots (bool): If True, save visualizations as PNG files instead of displaying.
        output_dir (str): Directory to save plots if save_plots is True; created if it doesn't exist.
    """
    for patient_dir in patient_dirs:
        if not os.path.exists(patient_dir):
            print(f"Error: Directory {patient_dir} does not exist.")
            continue
        
        print(f"\nProcessing patient directory: {patient_dir}")
        npz_files = [f for f in os.listdir(patient_dir) if f.endswith(".npz")]
        npz_files.sort()  # Sort for consistent order (e.g., slice_0.npz, slice_1.npz, ...)
        
        if not npz_files:
            print(f"No .npz files found in {patient_dir}")
            continue
        
        for i, file in enumerate(npz_files[:max_files_per_patient]):
            npz_file_path = os.path.join(patient_dir, file)
            inspect_npz(npz_file_path, save_plots, output_dir)
            if i + 1 == max_files_per_patient:
                print(f"Reached limit of {max_files_per_patient} files for {patient_dir}")
                break    
# Example usage
if __name__ == "__main__":
    # Specify base directory and example .npz file to inspect
    base_dir = r"E:\Study\IIT_Bombay\SEM_3\IE643_DeepLearning\Project\ie643_project\Data\Sample\Output"

    # Example: inspect a single .npz file
    npz_file = os.path.join(base_dir, "swin_mae", "BraTS2021_00000_meta.npz")

    # Optional: Directory to save plots (will be created if it doesn't exist)
    save_plots = False  # Set to True to save plots instead of displaying
    output_dir = os.path.join(base_dir, "plots")  # Changed from 'visualizations' to 'plots'

    # Call inspect_npz with a single file path (string), not a list
    inspect_npz(npz_file, save_plots=save_plots, output_dir=None)

    # If you want to inspect multiple .npz files inside patient directories, use inspect_multiple_npz
    # Example:
    # patient_dirs = [os.path.join(base_dir, "test", "patient_BraTS2021_00033")]
    # inspect_multiple_npz(patient_dirs, max_files_per_patient=133, save_plots=save_plots, output_dir=output_dir)