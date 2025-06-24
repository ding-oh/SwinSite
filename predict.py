import os
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader

from openbabel import pybel, openbabel
from scipy.spatial.distance import cdist

from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label

from tqdm.auto import tqdm

from proteindata import proteinDataset_predict
from SwinUnet import SwinSite
from utils import (
    prepare_data,
    load_multiple_models,
    ensemble_inference,
    save_pocket_mol,
    save_grid,
    save_molecule_files
)

def main(input_dir, file_format="mol2", model_paths=None, output_root="./output", log_path="./logs/log.txt", use_ensemble=True):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w")
    log_file.write("==== Prediction Log Start ====\n\n")

    if use_ensemble:
        models = load_multiple_models(model_paths)
    else:
        models = load_multiple_models(["./model/fold_2/best_epoch.h5"])

    os.makedirs(output_root, exist_ok=True)
    input_name = os.path.basename(os.path.normpath(input_dir))
    failed_count = 0

    try:
        log_file.write(f"[INFO] Start data preparation: {input_name}\n")
        print(f"Start data preparation: {input_name}")
        val_Dataloader = prepare_data(input_dir, file_format)
    except Exception as e:
        error_msg = f"[ERROR] Failed to prepare data for {input_name}: {e}\n"
        print(error_msg)
        log_file.write(error_msg)
        log_file.close()
        return

    output_path = os.path.join(output_root, input_name)
    os.makedirs(output_path, exist_ok=True)

    log_file.write(f"[INFO] Start prediction: {input_name}\n")
    print(f"Start prediction: {input_name}")

    for i, data in enumerate(tqdm(val_Dataloader, desc=f"{input_name} prediction")):
        try:
            input, origin, step, name, mol = data
            try:
                mol = next(pybel.readfile(file_format, mol[0]))
            except Exception as e:
                error_msg = f"[ERROR] Failed to read molecule file {mol[0]}: {e}\n"
                print(error_msg)
                log_file.write(error_msg)
                failed_count += 1
                continue

            input = input.cuda()

            try:
                output = ensemble_inference(models, input)
            except Exception as e:
                error_msg = f"[ERROR] Model inference failed for {name}: {e}\n"
                print(error_msg)
                log_file.write(error_msg)
                log_file.write(traceback.format_exc() + "\n")
                failed_count += 1
                continue

            try:
                pockets, binding_score = save_pocket_mol(output, origin[0], step[0], mol)
                pocket_grids, _ = save_grid(output, origin[0], step[0], mol)
            except Exception as e:
                error_msg = f"[ERROR] Failed to process pockets for {name}: {e}\n"
                print(error_msg)
                log_file.write(error_msg)
                log_file.write(traceback.format_exc() + "\n")
                failed_count += 1
                continue

            folder_name = os.path.join(output_path, str(name).split("'")[1])
            os.makedirs(folder_name, exist_ok=True)

            try:
                save_molecule_files(folder_name, pockets, "pocket", binding_score, file_format)
                save_molecule_files(folder_name, pocket_grids, "grid", binding_score, file_format)
            except Exception as e:
                error_msg = f"[ERROR] Failed to save molecule files for {name}: {e}\n"
                print(error_msg)
                log_file.write(error_msg)
                log_file.write(traceback.format_exc() + "\n")
                failed_count += 1
                continue

        except Exception as e:
            error_msg = f"[ERROR] Unexpected error in processing {name}: {e}\n"
            print(error_msg)
            log_file.write(error_msg)
            log_file.write(traceback.format_exc() + "\n")
            failed_count += 1
            continue

    summary_msg = f"==> Finished processing {input_name}, Failed samples: {failed_count}\n"
    print(summary_msg)
    log_file.write(summary_msg)
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protein Pocket Prediction")
    parser.add_argument("-i", "--input_dir", required=True, help="Path to input folder")
    parser.add_argument("-f", "--file_format", default="mol2", help="File format (default: mol2)")
    parser.add_argument("-m", "--model_paths", nargs='+', default=[
        "./model/fold_1/best_epoch.h5",
        "./model/fold_2/best_epoch.h5",
        "./model/fold_3/best_epoch.h5",
        "./model/fold_4/best_epoch.h5",
    ], help="List of model checkpoint paths")
    parser.add_argument("-o", "--output_root", default="./output", help="Root output directory")
    parser.add_argument("-l", "--log_path", default="./logs/log.txt", help="Log file path")
    parser.add_argument("--no_ensemble", action="store_true", help="Use single model instead of ensemble (default: ensemble)")

    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        file_format=args.file_format,
        model_paths=args.model_paths,
        output_root=args.output_root,
        log_path=args.log_path,
        use_ensemble=not args.no_ensemble
    )
