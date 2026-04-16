import os
from collections import OrderedDict
from math import ceil

import numpy as np
import torch
import h5py
from scipy.spatial.distance import cdist

from openbabel import pybel, openbabel
from skimage.segmentation import clear_border
from skimage.morphology import closing
from skimage.measure import label

from proteindata import Featurizer
from SwinUnet import SwinSite

# ========================================
# Constants
# ========================================
MAX_DIST = 35.0
SCALE = 0.66
RESOLUTION = 1.0 / SCALE
LARGE_THRESHOLD = 2 * MAX_DIST  # 70A
SIGMA = 1.0


# ========================================
# Model loading
# ========================================
def load_model(model_path, device='cuda'):
    """Load a single model from .h5 or .pth checkpoint."""
    model = SwinSite(in_channel=18, hidden_dim=96, num_classes=1, window_size=[3, 3, 3]).to(device)

    if model_path.endswith(".h5"):
        state_dict = OrderedDict()
        with h5py.File(model_path, 'r') as f:
            for key in f.keys():
                state_dict[key] = torch.from_numpy(f[key][()])
    else:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
        else:
            state_dict = ckpt

    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_ensemble(model_paths, device='cuda'):
    """Load multiple models for ensemble inference."""
    return [load_model(path, device) for path in model_paths]


def ensemble_inference(models, input_tensor):
    """Run ensemble inference (mean of model outputs)."""
    with torch.no_grad():
        outputs = [model(input_tensor) for model in models]
        return torch.mean(torch.stack(outputs, dim=0), dim=0)


# ========================================
# Grid construction
# ========================================
def gaussian_weight(distance, sigma):
    return np.exp(-(distance ** 2) / (2 * sigma ** 2))


def make_grid(coords, features, grid_resolution=1.0, max_dist=10.0, sigma=None):
    """Voxelize atom features onto a 3D grid.

    If sigma is given, applies Gaussian smoothing (matches ensemble training).
    Otherwise uses nearest-voxel assignment.
    """
    coords = np.asarray(coords, dtype=np.float64)
    features = np.asarray(features, dtype=np.float64)
    N, F = features.shape
    box_size = ceil(2 * max_dist / grid_resolution + 1)
    grid = np.zeros((box_size, box_size, box_size, F), dtype=np.float32)

    if sigma is not None and sigma > 0:
        for atom_coord, feat in zip(coords, features):
            atom_grid_coord = (atom_coord + max_dist) / grid_resolution
            window = int(np.ceil(3 * sigma / grid_resolution))
            lower = np.maximum(np.floor(atom_grid_coord - window), 0).astype(int)
            upper = np.minimum(np.floor(atom_grid_coord + window) + 1, box_size).astype(int)
            for i in range(lower[0], upper[0]):
                for j in range(lower[1], upper[1]):
                    for k in range(lower[2], upper[2]):
                        vc = np.array([i, j, k]) * grid_resolution - max_dist + grid_resolution / 2
                        w = gaussian_weight(np.linalg.norm(vc - atom_coord), sigma)
                        grid[i, j, k, :] += feat * w
    else:
        grid_coords = (coords + max_dist) / grid_resolution
        grid_coords = grid_coords.round().astype(int)
        in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
        for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
            grid[x, y, z] += f

    return grid


# ========================================
# Pocket segmentation
# ========================================
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


def get_pockets_segmentation_global(density_3d, min_size=70, scale=0.66,
                                     initial_threshold=0.5, min_threshold=0.05,
                                     step_val=0.05, overlap_threshold=0.3):
    """Multi-threshold segmentation for merged global density (single channel)."""
    voxel_size = (1 / scale) ** 3
    threshold = initial_threshold
    final_label_image = np.zeros_like(density_3d, dtype=int)
    pocket_scores = {}
    current_label = 1

    while threshold >= min_threshold:
        bw = closing(density_3d > threshold)
        cleared = clear_border(bw)
        label_image, num_labels = label(cleared, return_num=True)

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
        threshold -= step_val

    return final_label_image, pocket_scores


# ========================================
# Pocket extraction
# ========================================
def extract_pockets(density, origin, step, mol, dist_cutoff=4.5,
                    expand_residue=False, use_global=False):
    """Extract pocket mols and grid mols from density."""
    if use_global:
        pockets_label, pocket_scores = get_pockets_segmentation_global(density)
    else:
        pockets_label, pocket_scores = get_pockets_segmentation(density)

    n_pockets = int(pockets_label.max())
    pocket_mols = OrderedDict()
    pocket_binding_scores = OrderedDict()
    grid_mols = OrderedDict()
    grid_scores = OrderedDict()

    if n_pockets == 0:
        return pocket_mols, pocket_binding_scores, grid_mols, grid_scores

    coords = np.array([a.coords for a in mol.atoms])
    atom2residue = np.array([a.residue.idx for a in mol.atoms])
    max_len = max(len(a.atoms) for a in mol.residues)
    residue2atom = np.array([
        [a.idx - 1 for a in r.atoms] + [-1] * (max_len - len(r.atoms))
        for r in mol.residues
    ])

    for pocket_label in range(1, n_pockets + 1):
        score = pocket_scores.get(pocket_label, 0)

        indices = np.argwhere(pockets_label == pocket_label).astype('float32')
        grid_coords = indices * np.asarray(step) + np.asarray(origin)
        if grid_coords.size == 0:
            continue
        obmol = openbabel.OBMol()
        for c in grid_coords:
            a = obmol.NewAtom()
            a.SetVector(float(c[0]), float(c[1]), float(c[2]))
        grid_mols[pocket_label] = pybel.Molecule(obmol)
        grid_scores[pocket_label] = score

        distance = cdist(coords, grid_coords)
        close_atoms = np.where((distance < dist_cutoff).any(axis=1))[0]
        if len(close_atoms) == 0:
            continue
        if expand_residue:
            residue_ids = np.unique(atom2residue[close_atoms])
            close_atoms = np.concatenate(residue2atom[residue_ids])

        pocket_mol = mol.clone
        atoms_to_del = set(range(len(pocket_mol.atoms))) - set(int(x) for x in close_atoms)
        pocket_mol.OBMol.BeginModify()
        for aidx in sorted(atoms_to_del, reverse=True):
            atom = pocket_mol.OBMol.GetAtom(aidx + 1)
            pocket_mol.OBMol.DeleteAtom(atom)
        pocket_mol.OBMol.EndModify()
        for atom in pocket_mol.atoms:
            atom.chain = 'A'

        pocket_mols[pocket_label] = pocket_mol
        pocket_binding_scores[pocket_label] = score

    return pocket_mols, pocket_binding_scores, grid_mols, grid_scores


# ========================================
# Surface mode
# ========================================
def is_large_protein(coords, threshold=LARGE_THRESHOLD):
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    span = bbox_max - bbox_min
    return np.any(span > threshold), span, bbox_min, bbox_max


def make_subgrid(all_coords, all_features, center, max_dist=MAX_DIST, scale=SCALE, sigma=SIGMA):
    """Create a feature grid centered at `center`."""
    relative = all_coords - center
    in_range = np.all(np.abs(relative) <= max_dist, axis=1)
    local_coords = relative[in_range]
    local_features = all_features[in_range]

    if len(local_coords) == 0:
        return None, None, None

    resolution = 1.0 / scale
    grid = make_grid(local_coords, local_features, resolution, max_dist, sigma=sigma)
    origin = center - max_dist
    step = np.array([resolution] * 3)
    return grid, origin, step


def farthest_point_sampling(points, min_spacing):
    """Select subset of points with at least min_spacing between them."""
    if len(points) == 0:
        return points
    selected = [0]
    min_dists = cdist(points[0:1], points)[0]

    while True:
        farthest_idx = np.argmax(min_dists)
        if min_dists[farthest_idx] < min_spacing:
            break
        selected.append(farthest_idx)
        new_dists = cdist(points[farthest_idx:farthest_idx + 1], points)[0]
        min_dists = np.minimum(min_dists, new_dists)

    return points[selected]


def generate_surface_centers(prot_coords, mol, spacing=30.0):
    """Sample grid centers from C-alpha positions using farthest-point sampling."""
    ca_coords = []
    for atom in mol.atoms:
        res = atom.OBAtom.GetResidue()
        if res and atom.OBAtom.GetType().strip() == 'CA':
            ca_coords.append(atom.coords)

    if not ca_coords:
        ca_coords = prot_coords[::10].tolist()

    ca_coords = np.array(ca_coords)
    return farthest_point_sampling(ca_coords, spacing)


def merge_predictions(predictions):
    """Merge local prediction grids into a single global grid using max-pooling."""
    all_origins = np.array([p[1] for p in predictions])
    all_ends = np.array([p[1] + np.array(p[0].shape) * p[2] for p in predictions])
    step = predictions[0][2]

    global_min = all_origins.min(axis=0)
    global_max = all_ends.max(axis=0)
    global_shape = np.ceil((global_max - global_min) / step).astype(int)

    global_density = np.zeros(global_shape, dtype=np.float32)

    for density, origin, s in predictions:
        offset = np.round((origin - global_min) / step).astype(int)
        d, h, w = density.shape
        ed = min(offset[0] + d, global_shape[0])
        eh = min(offset[1] + h, global_shape[1])
        ew = min(offset[2] + w, global_shape[2])
        sd, sh, sw = ed - offset[0], eh - offset[1], ew - offset[2]

        global_density[offset[0]:ed, offset[1]:eh, offset[2]:ew] = np.maximum(
            global_density[offset[0]:ed, offset[1]:eh, offset[2]:ew],
            density[:sd, :sh, :sw]
        )

    return global_density, global_min, step


# ========================================
# Core prediction
# ========================================
def predict_protein(mol, models, device='cuda', spacing=30.0,
                    max_dist=MAX_DIST, scale=SCALE, sigma=SIGMA):
    """
    Predict binding sites for a single protein using ensemble inference.

    Large proteins (>70A span) are automatically handled via multi-grid
    surface sampling with C-alpha farthest-point sampling.

    Returns (pocket_mols, pocket_scores, grid_mols, grid_scores, span)
    """
    if not isinstance(models, list):
        models = [models]

    featurizer = Featurizer(save_molecule_codes=False)
    prot_coords, prot_features = featurizer.get_features(mol)

    is_large, span, bbox_min, bbox_max = is_large_protein(prot_coords)
    if len(prot_coords) > 15000:
        is_large = False

    resolution = 1.0 / scale

    if not is_large:
        centroid = prot_coords.mean(axis=0)
        local_coords = prot_coords - centroid
        grid = make_grid(local_coords, prot_features, resolution, max_dist, sigma=sigma)

        x = torch.from_numpy(grid).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        with torch.no_grad():
            output = ensemble_inference(models, x) if len(models) > 1 else models[0](x)

        density = output.cpu().numpy()
        origin = centroid - max_dist
        step_arr = np.array([resolution] * 3)

        pm, ps, gm, gs = extract_pockets(density, origin, step_arr, mol)
        return pm, ps, gm, gs, span

    centers = generate_surface_centers(prot_coords, mol, spacing)

    predictions = []
    for center in centers:
        center = np.array(center, dtype=np.float64)
        grid, origin, step_arr = make_subgrid(prot_coords, prot_features, center,
                                               max_dist, scale, sigma=sigma)
        if grid is None:
            continue

        x = torch.from_numpy(grid).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        with torch.no_grad():
            output = ensemble_inference(models, x) if len(models) > 1 else models[0](x)

        density = output.cpu().numpy()
        while density.ndim > 3 and density.shape[0] == 1:
            density = np.squeeze(density, axis=0)

        predictions.append((density, origin, step_arr))

    if not predictions:
        return OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict(), span

    merged_density, global_origin, global_step = merge_predictions(predictions)
    pm, ps, gm, gs = extract_pockets(merged_density, global_origin, global_step, mol, use_global=True)

    return pm, ps, gm, gs, span


# ========================================
# File I/O
# ========================================
def save_molecule_files(folder, molecules, prefix, binding_scores, file_format):
    if isinstance(molecules, dict):
        for pocket_id, (mol, (_, score)) in enumerate(zip(molecules.values(), binding_scores.items())):
            filename = f"{folder}/{prefix}{pocket_id}_score_{score:.4f}.{file_format}"
            mol.write(file_format, filename, overwrite=True)
    else:
        for pocket_id, (mol, (_, score)) in enumerate(zip(molecules, binding_scores.items())):
            filename = f"{folder}/{prefix}{pocket_id}_score_{score:.4f}.{file_format}"
            mol.write(file_format, filename, overwrite=True)
