"""
SwinSite — Protein Binding Site Prediction

Predicts binding pockets using a 4-fold ensemble of 3D Swin Transformers.
Large proteins (>70A span) are automatically handled via multi-grid
surface sampling with C-alpha farthest-point sampling.

Usage:
    python predict.py -i <input_dir> -o <output_dir>
    python predict.py -i <input_dir> -o <output_dir> --cpu
    python predict.py -i <input_dir> -f mol2 -m model/fold_1/best_epoch.h5
"""

import os
import argparse
import traceback

import torch
from openbabel import pybel
from tqdm.auto import tqdm

from utils import load_ensemble, predict_protein, save_molecule_files

pybel.ob.obErrorLog.StopLogging()


def main(input_dir, file_format="pdb", output_format="mol2", model_paths=None,
         output_root="./output", log_path="./logs/log.txt",
         spacing=30.0, device='cuda'):

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    log_file.write("==== SwinSite Prediction Log ====\n\n")

    models = load_ensemble(model_paths, device=device)
    log_file.write(f"[INFO] Loaded {len(models)} model(s) on {device}\n")

    os.makedirs(output_root, exist_ok=True)
    input_name = os.path.basename(os.path.normpath(input_dir))
    output_path = os.path.join(output_root, input_name)
    os.makedirs(output_path, exist_ok=True)

    sample_dirs = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    log_file.write(f"[INFO] {input_name}: {len(sample_dirs)} samples\n")
    print(f"Processing {input_name}: {len(sample_dirs)} samples ({len(models)} model ensemble)")

    failed_count = 0

    for sample_name in tqdm(sample_dirs, desc=f"{input_name} prediction"):
        sample_dir = os.path.join(input_dir, sample_name)
        mol_path = os.path.join(sample_dir, f"protein.{file_format}")

        if not os.path.exists(mol_path):
            log_file.write(f"[WARN] No protein.{file_format} in {sample_name}\n")
            continue

        try:
            mol = next(pybel.readfile(file_format, mol_path))
        except Exception as e:
            log_file.write(f"[ERROR] Failed to read {mol_path}: {e}\n")
            failed_count += 1
            continue

        try:
            pm, ps, gm, gs, span = predict_protein(
                mol, models, device=device, spacing=spacing
            )

            log_file.write(f"[INFO] {sample_name}: span=[{span[0]:.0f},{span[1]:.0f},{span[2]:.0f}], "
                           f"pockets={len(pm)}\n")

            folder_name = os.path.join(output_path, sample_name)
            os.makedirs(folder_name, exist_ok=True)

            if pm:
                save_molecule_files(folder_name, pm, "pocket", ps, output_format)
                save_molecule_files(folder_name, gm, "grid", gs, output_format)
            else:
                log_file.write(f"[WARN] {sample_name}: No pockets detected\n")

        except Exception as e:
            error_msg = f"[ERROR] {sample_name}: {e}\n"
            log_file.write(error_msg)
            log_file.write(traceback.format_exc() + "\n")
            failed_count += 1

    summary = f"\n==> Finished {input_name}. Failed: {failed_count}/{len(sample_dirs)}\n"
    print(summary)
    log_file.write(summary)
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SwinSite - Protein Binding Site Prediction")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to input folder")
    parser.add_argument("-f", "--file_format", default="pdb", help="Input file format (default: pdb)")
    parser.add_argument("-of", "--output_format", default="mol2", help="Output file format (default: mol2)")
    parser.add_argument("-m", "--model_paths", nargs='+', default=[
        "./model/fold_1/best_epoch.h5",
        "./model/fold_2/best_epoch.h5",
        "./model/fold_3/best_epoch.h5",
        "./model/fold_4/best_epoch.h5",
    ], help="List of model checkpoint paths")
    parser.add_argument("-o", "--output_root", default="./output", help="Root output directory")
    parser.add_argument("-l", "--log_path", default="./logs/log.txt", help="Log file path")
    parser.add_argument("--spacing", type=float, default=30.0,
                        help="Min spacing for surface sampling centers (default: 30A)")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")

    args = parser.parse_args()

    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cpu':
        print("Running on CPU")

    main(
        input_dir=args.input_dir,
        file_format=args.file_format,
        output_format=args.output_format,
        model_paths=args.model_paths,
        output_root=args.output_root,
        log_path=args.log_path,
        spacing=args.spacing,
        device=device,
    )
