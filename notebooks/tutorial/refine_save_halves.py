import argparse
from pathlib import Path

import torch
import yaml

from ddw.fit_model import LitUnet3D
from ddw.refine_tomogram import _refine_single_tomogram
from ddw.utils.normalization import get_avg_model_input_mean_and_std
from ddw.utils.mrctools import save_mrc_data


def gpu_to_list(gpu):
    if gpu is None:
        return None
    if isinstance(gpu, int):
        return [gpu]
    return gpu


def main(config_path):
    config_path = Path(config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    shared = cfg["shared"]
    refine = cfg["refine_tomogram"]

    tomo0_file = Path(shared["tomo0_files"][0])
    tomo1_file = Path(shared["tomo1_files"][0])

    project_dir = Path(shared["project_dir"])
    output_dir = project_dir / "refined_tomograms_halves"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_checkpoint_file = Path(refine["model_checkpoint_file"])

    subtomo_size = int(shared["subtomo_size"])
    mw_angle = int(shared["mw_angle"])
    subtomo_overlap = int(refine.get("subtomo_overlap", 32))
    batch_size = int(refine.get("batch_size", 1))
    num_workers = int(shared.get("num_workers", 0))
    standardize_full_tomos = bool(shared.get("standardize_full_tomos", False))
    recompute_normalization = bool(refine.get("recompute_normalization", True))

    gpu = gpu_to_list(shared.get("gpu", None))
    device = "cpu" if gpu is None else f"cuda:{gpu[0]}"

    print("Config:", config_path)
    print("MW angle:", mw_angle)
    print("Tomo0:", tomo0_file)
    print("Tomo1:", tomo1_file)
    print("Checkpoint:", model_checkpoint_file)
    print("Output dir:", output_dir)
    print("Batch size:", batch_size)
    print("Device:", device)

    model = LitUnet3D.load_from_checkpoint(model_checkpoint_file).to(device).eval()

    with torch.no_grad():
        if recompute_normalization:
            loc, scale = get_avg_model_input_mean_and_std(
                tomo_file=tomo0_file,
                subtomo_size=subtomo_size,
                subtomo_extraction_strides=3 * [subtomo_size - subtomo_overlap],
                mw_angle=mw_angle,
                batch_size=batch_size,
                standardize=standardize_full_tomos,
                num_workers=num_workers,
                verbose=True,
            )
        else:
            loc = model.unet.normalization_loc.clone().detach().item()
            scale = model.unet.normalization_scale.clone().detach().item()

        refined0 = _refine_single_tomogram(
            tomo_file=tomo0_file,
            lightning_model=model,
            subtomo_size=subtomo_size,
            subtomo_overlap=subtomo_overlap,
            mw_angle=mw_angle,
            normalization_loc=loc,
            normalization_scale=scale,
            num_workers=num_workers,
            batch_size=batch_size,
            pbar_desc="Refining tomo0 / even",
        )

        refined1 = _refine_single_tomogram(
            tomo_file=tomo1_file,
            lightning_model=model,
            subtomo_size=subtomo_size,
            subtomo_overlap=subtomo_overlap,
            mw_angle=mw_angle,
            normalization_loc=loc,
            normalization_scale=scale,
            num_workers=num_workers,
            batch_size=batch_size,
            pbar_desc="Refining tomo1 / odd",
        )

    out0 = output_dir / f"{tomo0_file.stem}_mw{mw_angle}_refined{tomo0_file.suffix}"
    out1 = output_dir / f"{tomo1_file.stem}_mw{mw_angle}_refined{tomo1_file.suffix}"
    out_avg = output_dir / f"{tomo0_file.stem}+{tomo1_file.stem}_mw{mw_angle}_avg_refined{tomo0_file.suffix}"

    print("Saving refined tomo0/even to", out0)
    save_mrc_data(refined0.cpu(), str(out0), save=True)

    print("Saving refined tomo1/odd to", out1)
    save_mrc_data(refined1.cpu(), str(out1), save=True)

    print("Saving average to", out_avg)
    save_mrc_data(((refined0 + refined1) / 2).cpu(), str(out_avg), save=True)

    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
