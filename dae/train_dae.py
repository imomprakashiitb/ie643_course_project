from typing import Union, Optional
from collections import defaultdict
from functools import partial
from pathlib import Path
import numpy as np
import random
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt

from unet import UNet
from trainer import Trainer
from data_descriptor import BrainAEDataDescriptor, DataDescriptor
from utilities import ModelSaver, median_pool
from training import simple_train_step, simple_val_step
from metrics import Loss

def denoising(identifier: str, data: Optional[Union[str, DataDescriptor]] = None, lr=0.001, depth=4, wf=6, n_input=1, noise_std=0.2, noise_res=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def noise(x):
        ns = torch.normal(mean=torch.zeros(x.shape[0], x.shape[1], noise_res, noise_res), std=noise_std).to(x.device)
        ns = F.interpolate(ns, size=[128, 128], mode='bilinear', align_corners=False)
        roll_x = random.choice(range(128))
        roll_y = random.choice(range(128))
        ns = torch.roll(ns, shifts=[roll_x, roll_y], dims=[-2, -1])
        mask = x.sum(dim=1, keepdim=True) > 0.01
        ns *= mask
        res = x + ns
        return res

    def get_scores(trainer, batch, median_f=True):
        x = batch[0]
        trainer.model = trainer.model.eval()
        with torch.no_grad():
            clean = x.view(x.shape[0], -1, x.shape[-2], x.shape[-1]).clone().to(trainer.device)
            mask = clean.sum(dim=1, keepdim=True) > 0.01
            mask = (F.avg_pool2d(mask.float(), kernel_size=5, stride=1, padding=2) > 0.95)
            res = trainer.model(clean)
            err = ((clean - res) * mask).abs().mean(dim=1, keepdim=True)
            if median_f:
                err = median_pool(err, kernel_size=5, stride=1, padding=2)
        return err.cpu()

    def loss_f(trainer, batch, batch_results):
        mask = batch[0].sum(dim=1, keepdim=True) > 0.01  # Foreground mask only
        return (torch.pow(batch_results - batch[0], 2) * mask.float()).mean()

    def forward(trainer, batch):
        batch[1] = batch[0]  # Use input as target for denoising
        batch[0] = noise(batch[0].clone())
        return trainer.model(batch[0])

    model = UNet(in_channels=n_input, n_classes=n_input, norm="group", up_mode="upconv", depth=depth, wf=wf, padding=True).to(device)

    train_step = partial(simple_train_step, forward=forward, loss_f=loss_f)
    val_step = partial(simple_val_step, forward=forward, loss_f=loss_f)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=0.00001)
    callback_dict = defaultdict(list)

    model_saver = ModelSaver(path=Path("/content/drive/MyDrive/ie643_course_project_24M1644/saved_models") / f"{identifier}.pt")
    model_saver.register(callback_dict)

    Loss(lambda batch_res, batch: loss_f(trainer, batch, batch_res)).register(callback_dict, log=True, tensorboard=True, train=True, val=True)

    def visualize_progress(trainer, n_slices: int = 5, seed: int = None):

        """
        Save visualization of Clean, Noisy, Reconstructed, and Anomaly maps
        for n_slices fixed validation samples across all epochs.
        """
        trainer.model.eval()
        with torch.no_grad():
            device = trainer.device if hasattr(trainer, "device") else next(trainer.model.parameters()).device

            val_ds = trainer.val_dataloader.dataset
            ds_len = len(val_ds)
            if ds_len == 0:
                return

            # choose fixed absolute viz indices once
            if "viz_indices" not in trainer.state:
                if seed is not None:
                    random.seed(seed)
                k = min(n_slices, ds_len)
                trainer.state["viz_indices"] = random.sample(list(range(ds_len)), k)
                print(f"[visualize_progress] viz indices: {trainer.state['viz_indices']}")

            viz_indices = trainer.state["viz_indices"]
            k = len(viz_indices)

            # load slices by absolute index and stack into a batch on device
            inp_list = []
            for idx in viz_indices:
                item = val_ds[idx]
                x = item[0] if isinstance(item, (list, tuple)) else item
                if not isinstance(x, torch.Tensor):
                    x = torch.from_numpy(x)
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                inp_list.append(x)
            inp_b = torch.stack(inp_list, dim=0).to(device, non_blocking=True)  # (k, C, H, W)

            # forward / noise on device
            noisy_b = trainer.noise(inp_b.clone())
            recon_b = trainer.model(inp_b)

            # compute scores (expect tensor (k,1,H,W) or (k,H,W))
            tmp_batch = (inp_b, None)
            scores_t = trainer.get_scores(trainer, tmp_batch)
            scores_np = scores_t.squeeze().cpu().numpy()

            # set fixed vmax on first call (use maximum observed among chosen slices)
            if "viz_vmax" not in trainer.state:
                # protect against zero
                vmax0 = float(np.max(scores_np)) if np.max(scores_np) > 0 else 1.0
                trainer.state["viz_vmax"] = vmax0
                print(f"[visualize_progress] set viz_vmax = {trainer.state['viz_vmax']}")

            viz_vmax = float(trainer.state["viz_vmax"])
            # normalize to [0,1]
            scores_norm = np.clip(scores_np / (viz_vmax + 1e-12), 0.0, 1.0)

            # move images to CPU numpy for plotting
            inp_np = inp_b.detach().cpu().numpy()
            noisy_np = noisy_b.detach().cpu().numpy()
            recon_np = recon_b.detach().cpu().numpy()

            # plotting grid 4 x k
            out_dir = Path("/content/drive/MyDrive/ie643_course_project_24M1644/figures")
            out_dir.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(4, k, figsize=(4 * k, 12))
            if k == 1:
                axes = np.expand_dims(axes, axis=1)

            for col in range(k):
                clean_img = inp_np[col].squeeze()
                noisy_img = noisy_np[col].squeeze()
                recon_img = recon_np[col].squeeze()
                score_img = scores_norm[col]  # normalized 0..1

                axes[0, col].imshow(clean_img, cmap="gray"); axes[0, col].axis("off"); axes[0, col].set_title(f"Clean idx={viz_indices[col]}")
                axes[1, col].imshow(noisy_img, cmap="gray"); axes[1, col].axis("off"); axes[1, col].set_title("Noisy")
                axes[2, col].imshow(recon_img, cmap="gray"); axes[2, col].axis("off"); axes[2, col].set_title("Recon")
                im = axes[3, col].imshow(score_img, cmap="hot", vmin=0, vmax=1)  # fixed 0..1 scale
                axes[3, col].axis("off"); axes[3, col].set_title("Anomaly (0-1)")

            plt.tight_layout()
            fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
            epoch_no = trainer.state.get("epoch_no", 0)
            fname = out_dir / f"epoch_{epoch_no}_fixed5_norm01.png"
            plt.savefig(fname, bbox_inches="tight", dpi=150)
            plt.close(fig)

    trainer = Trainer(model=model,
                      train_dataloader=None,
                      val_dataloader=None,
                      optimiser=optimiser,
                      train_step=train_step,
                      val_step=val_step,
                      callback_dict=callback_dict,
                      device=device,
                      identifier=identifier)

    trainer.noise = noise
    trainer.get_scores = get_scores
    trainer.set_data(data)
    trainer.reset_state()

    trainer.lr_scheduler = CosineAnnealingLR(optimizer=optimiser, T_max=200)
    trainer.callback_dict["after_train_epoch"].append(lambda trainer: trainer.lr_scheduler.step())
    trainer.callback_dict["after_val_epoch"].append(visualize_progress)

    return trainer

def train(id: str = "model", noise_res: int = 16, noise_std: float = 0.2, seed: int = 0, batch_size: int = 16, slice_percentage: float = 0.5):
    dd = BrainAEDataDescriptor(dataset="ixi_braTS",
                               n_train_patients=None,
                               n_val_patients=None,
                               seed=seed,
                               batch_size=batch_size,
                               data_path="/content/drive/MyDrive/ie643_course_project_24M1644/dae_input_data")

    # Randomly select percentage of slices per MRI (but avoid double-sampling)
    train_dataset = dd.get_dataset("train")

    # Detect whether PatientDataset already applied slice_percentage internally.
    # If any patient has slice_percentage < 1.0, assume pre-sampling already happened.
    pre_sampled = any(getattr(p, "slice_percentage", 1.0) < 1.0 for p in train_dataset.patient_datasets)

    if pre_sampled:
        print("Detected pre-sampling inside PatientDataset (slice_percentage < 1.0). Skipping extra sampling in train().")
    else:
        # Sample LOCAL indices per patient (works with current update_slice_selection which expects local indices)
        selected_slices = {}
        for pid, patient_dataset in enumerate(train_dataset.patient_datasets):
            n_slices = len(patient_dataset)
            # pick number of slices to sample: at least 1 (use ceil to avoid rounding down too aggressively)
            k = max(1, int(math.ceil(n_slices * slice_percentage)))
            k = min(k, n_slices)
            local_indices = list(range(n_slices))
            chosen = random.sample(local_indices, k)
            chosen.sort()
            selected_slices[pid] = chosen

        # Apply selection (this expects LOCAL indices)
        train_dataset.update_slice_selection(selected_slices)
        total_after = sum(len(p) for p in train_dataset.patient_datasets)
        print(f"Applied sampling in train(): patients={len(train_dataset.patient_datasets)}, total slices after selection={total_after}")
        print(f"ConcatDataset length (dataset): {len(train_dataset.dataset)}")

    trainer = denoising(id, data=dd, lr=0.0001, depth=4, wf=6, noise_std=noise_std, noise_res=noise_res, n_input=1)
    checkpoint_path = Path("/content/drive/MyDrive/ie643_course_project_24M1644/saved_models") / f"{id}_checkpoint.pt"
    start_epoch = 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimiser.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        trainer.state['epoch_no'] = start_epoch
        print(f"Resumed from checkpoint at epoch {start_epoch}")
    trainer.train(max_epochs=200, epoch_len=32, val_epoch_len=32)
    torch.save({
        'epoch': trainer.state['epoch_no'],
        'model_state_dict': trainer.model.state_dict(),
        'optimizer_state_dict': trainer.optimiser.state_dict(),
        'scheduler_state_dict': trainer.lr_scheduler.state_dict(),
    }, checkpoint_path)
    print(f"Final checkpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--identifier", type=str, default="daemodel", help="model name")
    parser.add_argument("-nr", "--noise_res", type=int, default=16, help="noise resolution")
    parser.add_argument("-ns", "--noise_std", type=float, default=0.2, help="noise magnitude")
    parser.add_argument("-s", "--seed", type=int, default=0, help="random seed")
    parser.add_argument("-bs", "--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("-sp", "--slice_percentage", type=float, default=0.5, help="percentage of slices per MRI")
    args = parser.parse_args()
    train(id=args.identifier, noise_res=args.noise_res, noise_std=args.noise_std, seed=args.seed,
          batch_size=args.batch_size, slice_percentage=args.slice_percentage)