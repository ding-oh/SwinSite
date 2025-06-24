import os
import torch
import numpy as np
import h5py
from collections import OrderedDict
from torch.utils.data import DataLoader
from openbabel import pybel, openbabel
from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label
from scipy.spatial.distance import cdist

from proteindata import proteinDataset_predict
from SwinUnet import SwinSite

def prepare_data(input_file, file_format):
    val_dataset = proteinDataset_predict(data_path=input_file, file_format=file_format, scale=0.66)
    return DataLoader(val_dataset, batch_size=1)

def load_multiple_models(model_paths):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    models = []

    for model_path in model_paths:
        model = SwinSite(in_channel=18, hidden_dim=96, num_classes=1, window_size=[3, 3, 3]).to(device)

        if model_path.endswith(".h5"):
            state_dict = {}
            with h5py.File(model_path, 'r') as f:
                for key in f.keys():
                    state_dict[key] = torch.tensor(f[key][:])
        else:
            checkpoint = torch.load(model_path, map_location=torch.device(device))
            state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)

    return models


def ensemble_inference(models, input_tensor, method="mean"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = [model(input_tensor) for model in models]
        stacked_outputs = torch.stack(outputs, dim=2)

        if method == "mean":
            final_output = torch.mean(stacked_outputs, dim=2)
        elif method == "max":
            final_output, _ = torch.max(stacked_outputs, dim=2)
        else:
            raise ValueError("Invalid method. Choose 'mean' or 'max'.")

    return final_output.cpu()

def get_pockets_segmentation(density, initial_threshold=0.5, min_size=70, scale=0.66, 
                             min_threshold=0.05, step=0.05, overlap_threshold=0.3):
    voxel_size = (1 / scale) ** 3
    threshold = initial_threshold
    final_label_image = np.zeros_like(density[0, 0], dtype=int)
    pocket_scores = {}
    current_label = 1

    while threshold >= min_threshold:
        bw = closing((density[0] > threshold).any(axis=0))
        cleared = clear_border(bw)
        label_image, num_labels = label(cleared, return_num=True)
        density_3d = density[0, 0]

        for i in range(1, num_labels + 1):
            pocket_idx = (label_image == i)
            pocket_size = pocket_idx.sum() * voxel_size
            if pocket_size < min_size:
                continue

            overlap = (final_label_image > 0) & pocket_idx
            overlap_ratio = overlap.sum() / pocket_idx.sum()
            if overlap_ratio > overlap_threshold:
                continue

            final_label_image[pocket_idx & (final_label_image == 0)] = current_label
            pocket_scores[current_label] = density_3d[pocket_idx].mean().item()
            current_label += 1

        threshold -= step

    return final_label_image, pocket_scores

def segment_density(density, mol, step, origin):
    coords = np.array([a.coords for a in mol.atoms])
    atom2residue = np.array([a.residue.idx for a in mol.atoms])
    max_len = max(len(a.atoms) for a in mol.residues)
    residue2atom = np.array([a_list + [-1] * (max_len - len(a_list)) 
                             for a_list in [[a.idx - 1 for a in r.atoms] for r in mol.residues]])
    pockets, pocket_scores = get_pockets_segmentation(density)
    return coords, atom2residue, residue2atom, pockets, pocket_scores

def save_pocket_mol(density, origin, step, mol, dist_cutoff=4.5, expand_residue=False, chain_id='A'):
    coords, atom2residue, residue2atom, pockets, pocket_scores = segment_density(density, mol, step, origin)
    pocket_mols = []
    pocket_binding_scores = {}

    for pocket_label in range(1, pockets.max() + 1):
        indices = np.argwhere(pockets == pocket_label).astype('float32')
        indices *= np.asarray(step)
        indices += np.asarray(origin)
        distance = cdist(coords, indices)
        close_atoms = np.where((distance < dist_cutoff).any(axis=1))[0]
        if len(close_atoms) == 0:
            continue
        if expand_residue:
            residue_ids = np.unique(atom2residue[close_atoms])
            close_atoms = np.concatenate(residue2atom[residue_ids])

        pocket_mol = mol.clone
        atoms_to_del = (set(range(len(pocket_mol.atoms))) - set(close_atoms))
        pocket_mol.OBMol.BeginModify()
        for aidx in sorted(atoms_to_del, reverse=True):
            atom = pocket_mol.OBMol.GetAtom(aidx + 1)
            pocket_mol.OBMol.DeleteAtom(atom)
        pocket_mol.OBMol.EndModify()

        for atom in pocket_mol.atoms:
            atom.chain = chain_id

        pocket_mols.append(pocket_mol)
        pocket_binding_scores[pocket_label] = pocket_scores.get(pocket_label, 0)

    return pocket_mols, pocket_binding_scores

def save_grid(density, origin, step, mol):
    coords, atom2residue, residue2atom, pockets, binding_score = segment_density(density, mol, step, origin)
    pocket_grid = []

    for pocket_label in range(1, pockets.max() + 1):
        indices = np.argwhere(pockets == pocket_label).astype('float32')
        indices *= np.asarray(step)
        indices += np.asarray(origin)
        if indices.size == 0:
            continue
        mol = openbabel.OBMol()
        for idx in indices:
            a = mol.NewAtom()
            a.SetVector(float(idx[0]), float(idx[1]), float(idx[2]))
        p_mol = pybel.Molecule(mol)
        pocket_grid.append(p_mol)

    return pocket_grid, binding_score

def save_molecule_files(folder, molecules, prefix, binding_scores, file_format):
    for pocket_id, (mol, (_, score)) in enumerate(zip(molecules, binding_scores.items())):
        filename = f"{folder}/{prefix}{pocket_id}_score_{score:.4f}.{file_format}"
        mol.write(file_format, filename, overwrite=True)
