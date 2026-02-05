import argparse
from pathlib import Path

import numpy as np
import pyrootutils
import torch
from PIL import Image
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True, dotenv=True)

from common.data.hdf5_datasets import HDF5Dataset_for_MotoGPT_CALVINLike  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Collect samples with identical current and future frames.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="/data/250010208/whl/code/Moto/latent_motion_tokenizer/configs/train/train_aloha.yaml",
        help="Path to the training configuration that defines dataset and dataloader settings.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./identical_samples",
        help="Directory where the visualizations will be saved.",
    )
    parser.add_argument(
        "--max_visualizations",
        type=int,
        default=50,
        help="Maximum number of qualifying samples to visualize.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Dataset split to inspect.",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1,
        help="Number of future frames to load per sample.",
    )
    return parser.parse_args()


def frames_match(initial_tensor, future_tensor, mask_tensor):
    valid_indices = torch.nonzero(mask_tensor > 0, as_tuple=False).squeeze(-1)
    if valid_indices.numel() == 0:
        return False, None
    reference = initial_tensor[0]
    expanded_reference = reference.unsqueeze(0).expand_as(future_tensor[valid_indices])
    static_match = torch.equal(future_tensor[valid_indices], expanded_reference)
    if not static_match:
        return False, None
    return True, valid_indices


def actions_constant(actions_tensor, mask_tensor, atol=1e-6):
    valid_mask = mask_tensor > 0
    if not valid_mask.any():
        return False
    valid_actions = actions_tensor[valid_mask]
    reference = valid_actions[0]
    max_deviation = torch.max(torch.abs(valid_actions - reference))
    return max_deviation <= atol


def save_collage(sample, output_path):
    images = [
        sample["static_initial"],
        sample["static_future"],
        sample["gripper_initial"],
        sample["gripper_future"],
    ]
    pil_images = []
    for tensor in images:
        array = tensor.cpu().numpy()
        if array.dtype != np.uint8:
            array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
        array = np.transpose(array, (1, 2, 0))
        pil_images.append(Image.fromarray(array))
    width, height = pil_images[0].size
    canvas = Image.new("RGB", (width * 2, height * 2))
    canvas.paste(pil_images[0], (0, 0))
    canvas.paste(pil_images[1], (width, 0))
    canvas.paste(pil_images[2], (0, height))
    canvas.paste(pil_images[3], (width, height))
    canvas.save(output_path)


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    cfg = OmegaConf.load(args.config_path)
    dataset_cfg = OmegaConf.load(cfg["dataset_config_path"]) if isinstance(cfg["dataset_config_path"], str) else cfg["dataset_config_path"]

    dataset = HDF5Dataset_for_MotoGPT_CALVINLike(
        hdf5_dir=dataset_cfg["hdf5_dir"],
        split=args.split,
        skip_frame=dataset_cfg["skip_frame"],
        sequence_length=args.sequence_length,
        do_extract_future_frames=True,
        do_extract_action=True,
        rgb_shape=tuple(dataset_cfg["rgb_shape"]),
        chunk_size=30
    )

    dataloader_cfg = cfg.get("dataloader_config", {})
    num_workers = int(dataloader_cfg.get("workers_per_gpu", 0))
    batch_size = int(dataloader_cfg.get("bs_per_gpu", 1))
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        if "prefetch_factor" in dataloader_cfg:
            loader_kwargs["prefetch_factor"] = int(dataloader_cfg["prefetch_factor"])
    dataloader = DataLoader(dataset, **loader_kwargs)

    total_samples = 0
    matching_samples = 0
    selected_samples = []

    for batch in tqdm(dataloader, total=len(dataloader), desc="Scanning", unit="batch"):
        batch_size = batch["rgb_initial_static"].shape[0]
        total_samples += batch_size

        for i in range(batch_size):
            latent_mask = batch["latent_mask"][i]
            # static_match, valid_indices = frames_match(
            #     batch["rgb_initial_static"][i],
            #     batch["rgb_future_static"][i],
            #     latent_mask,
            # )
            # if not static_match:
            #     continue
            # gripper_match, _ = frames_match(
            #     batch["rgb_initial_gripper"][i],
            #     batch["rgb_future_gripper"][i],
            #     latent_mask,
            # )
            # if not gripper_match:
            #     continue
            actions_match = actions_constant(
                batch["actions"][i],
                batch["mask"][i],
            )
            if not actions_match:
                continue
            print("Found identical frames sample")

            matching_samples += 1
            if len(selected_samples) < args.max_visualizations:
                first_valid = 0
                selected_samples.append(
                    {
                        "static_initial": batch["rgb_initial_static"][i, 0].cpu().clone(),
                        "static_future": batch["rgb_future_static"][i, first_valid].cpu().clone(),
                        "gripper_initial": batch["rgb_initial_gripper"][i, 0].cpu().clone(),
                        "gripper_future": batch["rgb_future_gripper"][i, first_valid].cpu().clone(),
                        "dataset_idx": int(batch["idx"][i]),
                    }
                )

    proportion = matching_samples / total_samples if total_samples else 0.0
    print(f"Total samples: {total_samples}")
    print(f"Identical samples: {matching_samples}")
    print(f"Proportion: {proportion:.6f}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for rank, sample in enumerate(selected_samples):
        filename = f"identical_sample_{rank:03d}_idx_{sample['dataset_idx']}.png"
        output_path = output_dir / filename
        save_collage(sample, output_path)

    print(f"Saved {len(selected_samples)} visualizations to {output_dir}")


if __name__ == "__main__":
    main()
