from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn.functional as F
import numpy as np
import openbabel
from openbabel import pybel
import os
import pickle
from math import ceil, sin, cos, sqrt, pi
from itertools import combinations
from random import choice
from skimage.draw import ellipsoid
from scipy import ndimage

# Disable openbabel error logging.
pybel.ob.obErrorLog.StopLogging()

#############################################
# Featurizer 클래스: 원자 기반 피처 추출 구현
#############################################
class Featurizer():
    def __init__(self, atom_codes=None, atom_labels=None,
                 named_properties=None, save_molecule_codes=True,
                 custom_properties=None, smarts_properties=None,
                 smarts_labels=None):

        self.FEATURE_NAMES = []

        if atom_codes is not None:
            if not isinstance(atom_codes, dict):
                raise TypeError('Atom codes should be dict, got %s instead'
                                % type(atom_codes))
            codes = set(atom_codes.values())
            for i in range(len(codes)):
                if i not in codes:
                    raise ValueError('Incorrect atom code %s' % i)
            self.NUM_ATOM_CLASSES = len(codes)
            self.ATOM_CODES = atom_codes
            if atom_labels is not None:
                if len(atom_labels) != self.NUM_ATOM_CLASSES:
                    raise ValueError('Incorrect number of atom labels: %s instead of %s'
                                     % (len(atom_labels), self.NUM_ATOM_CLASSES))
            else:
                atom_labels = ['atom%s' % i for i in range(self.NUM_ATOM_CLASSES)]
            self.FEATURE_NAMES += atom_labels
        else:
            self.ATOM_CODES = {}
            metals = ([3, 4, 11, 12, 13] + list(range(19, 32))
                      + list(range(37, 51)) + list(range(55, 84))
                      + list(range(87, 104)))
            atom_classes = [
                (5, 'B'),
                (6, 'C'),
                (7, 'N'),
                (8, 'O'),
                (15, 'P'),
                (16, 'S'),
                (34, 'Se'),
                ([9, 17, 35, 53], 'halogen'),
                (metals, 'metal')
            ]
            for code, (atom, name) in enumerate(atom_classes):
                if isinstance(atom, list):
                    for a in atom:
                        self.ATOM_CODES[a] = code
                else:
                    self.ATOM_CODES[atom] = code
                self.FEATURE_NAMES.append(name)
            self.NUM_ATOM_CLASSES = len(atom_classes)

        if named_properties is not None:
            if not isinstance(named_properties, (list, tuple, np.ndarray)):
                raise TypeError('named_properties must be a list')
            allowed_props = [prop for prop in dir(pybel.Atom) if not prop.startswith('__')]
            for prop_id, prop in enumerate(named_properties):
                if prop not in allowed_props:
                    raise ValueError('named_properties must be in pybel.Atom attributes, %s was given at position %s' % (prop, prop_id))
            self.NAMED_PROPS = named_properties
        else:
            # 기본적으로 사용할 pybel.Atom 속성
            self.NAMED_PROPS = ['hyb', 'heavydegree', 'heterodegree', 'partialcharge']
        self.FEATURE_NAMES += self.NAMED_PROPS

        if not isinstance(save_molecule_codes, bool):
            raise TypeError('save_molecule_codes should be bool, got %s instead' % type(save_molecule_codes))
        self.save_molecule_codes = save_molecule_codes
        if save_molecule_codes:
            self.FEATURE_NAMES.append('molcode')

        self.CALLABLES = []
        if custom_properties is not None:
            for i, func in enumerate(custom_properties):
                if not callable(func):
                    raise TypeError('custom_properties should be list of callables, got %s instead' % type(func))
                name = getattr(func, '__name__', '')
                if name == '':
                    name = 'func%s' % i
                self.CALLABLES.append(func)
                self.FEATURE_NAMES.append(name)

        if smarts_properties is None:
            self.SMARTS = [
                '[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]',
                '[a]',
                '[!$([#1,#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]',
                '[!$([#6,H0,-,-2,-3]),$([!H0;#7,#8,#9])]',
                '[r]'
            ]
            smarts_labels = ['hydrophobic', 'aromatic', 'acceptor', 'donor', 'ring']
        elif not isinstance(smarts_properties, (list, tuple, np.ndarray)):
            raise TypeError('smarts_properties must be a list')
        else:
            self.SMARTS = smarts_properties

        if smarts_labels is not None:
            if len(smarts_labels) != len(self.SMARTS):
                raise ValueError('Incorrect number of SMARTS labels: %s instead of %s' % (len(smarts_labels), len(self.SMARTS)))
        else:
            smarts_labels = ['smarts%s' % i for i in range(len(self.SMARTS))]

        self.compile_smarts()
        self.FEATURE_NAMES += smarts_labels

    def compile_smarts(self):
        self.__PATTERNS = []
        for smarts in self.SMARTS:
            self.__PATTERNS.append(pybel.Smarts(smarts))

    def encode_num(self, atomic_num):
        if not isinstance(atomic_num, int):
            raise TypeError('Atomic number must be int, %s was given' % type(atomic_num))
        encoding = np.zeros(self.NUM_ATOM_CLASSES)
        try:
            encoding[self.ATOM_CODES[atomic_num]] = 1.0
        except:
            pass
        return encoding

    def find_smarts(self, molecule):
        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given' % type(molecule))
        features = np.zeros((len(molecule.atoms), len(self.__PATTERNS)))
        for (pattern_id, pattern) in enumerate(self.__PATTERNS):
            atoms_with_prop = np.array(list(*zip(*pattern.findall(molecule))), dtype=int) - 1
            features[atoms_with_prop, pattern_id] = 1.0
        return features

    def get_binary_features(self, mol):
        coords = [a.coords for a in mol.atoms]
        coords = np.array(coords)
        features = np.ones((len(coords), 1))
        return coords, features

    def get_features(self, molecule, molcode=None):
        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given' % type(molecule))
        if molcode is None and self.save_molecule_codes:
            raise ValueError('save_molecule_codes is set to True, you must specify code for the molecule')
        coords = []
        features = []
        heavy_atoms = []
        for i, atom in enumerate(molecule):
            if atom.atomicnum > 1:
                heavy_atoms.append(i)
                coords.append(atom.coords)
                features.append(np.concatenate((
                    self.encode_num(atom.atomicnum),
                    [atom.__getattribute__(prop) for prop in self.NAMED_PROPS],
                    [func(atom) for func in self.CALLABLES],
                )))
        coords = np.array(coords, dtype=np.float32)
        features = np.array(features, dtype=np.float32)
        if self.save_molecule_codes:
            features = np.hstack((features, molcode * np.ones((len(features), 1))))
        features = np.hstack([features, self.find_smarts(molecule)[heavy_atoms]])
        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')
        return coords, features

    def get_features_gt(self, molecule, molcode=None):
        if not isinstance(molecule, pybel.Molecule):
            raise TypeError('molecule must be pybel.Molecule object, %s was given' % type(molecule))
        if molcode is None and self.save_molecule_codes:
            raise ValueError('save_molecule_codes is set to True, you must specify code for the molecule')
        pocket_coords, pocket_features = self.get_binary_features(molecule)
        features = pocket_features
        coords = pocket_coords
        if np.isnan(features).any():
            raise RuntimeError('Got NaN when calculating features')
        return coords, features

    def to_pickle(self, fname='featurizer.pkl'):
        patterns = self.__PATTERNS[:]
        del self.__PATTERNS
        try:
            with open(fname, 'wb') as f:
                pickle.dump(self, f)
        finally:
            self.__PATTERNS = patterns[:]

    @staticmethod
    def from_pickle(fname):
        with open(fname, 'rb') as f:
            featurizer = pickle.load(f)
        featurizer.compile_smarts()
        return featurizer

#############################################
# proteinDataset 클래스: Training/Evaluation용 데이터셋
# - 원자 좌표를 격자로 바꾸는 과정에 Gaussian smoothing 옵션 포함
#############################################
class proteinDataset(Dataset):
    def __init__(self, data_path, featurizer=Featurizer(save_molecule_codes=False),
                 max_dist=35, eval=True, scale=1, max_translation=5, kfold_ind=0,
                 use_gaussian=True, sigma=1.0):
        self.data_path = data_path
        self.max_dist = max_dist
        self.scale = scale
        self.max_translation = max_translation
        self.eval = eval
        self.featurizer = featurizer
        self.use_gaussian = use_gaussian   # Gaussian smoothing 사용 여부
        self.sigma = sigma                 # Gaussian 표준편차 (예: grid_resolution=1.0일 때 1.0 Å)
        self.data_list = sorted([os.path.join(self.data_path, x) for x in os.listdir(self.data_path)])
        self.eval_num = int(len(self.data_list) / 4)
        footprint = ellipsoid(2, 2, 2)
        self.footprint = footprint.reshape((*footprint.shape, 1))
        self.data_list_list = [self.data_list[:self.eval_num],
                               self.data_list[self.eval_num:self.eval_num*2],
                               self.data_list[self.eval_num*2:self.eval_num*3],
                               self.data_list[-self.eval_num:]]
        if self.eval:
            self.data_list = self.data_list_list[kfold_ind]
            self.protein_list = [next(pybel.readfile('mol2', os.path.join(path, 'protein.mol2')))
                                 for path in self.data_list]
            self.pocket_list = [next(pybel.readfile('mol2', os.path.join(path, 'cavity6.mol2')))
                                for path in self.data_list]
        else:
            self.data_list = self.data_list_list[0] + self.data_list_list[1] + \
                             self.data_list_list[2] + self.data_list_list[3]
            self.protein_list = [next(pybel.readfile('mol2', os.path.join(path, 'protein.mol2')))
                                 for path in self.data_list]
            self.pocket_list = [next(pybel.readfile('mol2', os.path.join(path, 'cavity6.mol2')))
                                for path in self.data_list]

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        path = self.data_list[index]
        mol1 = self.protein_list[index]
        mol2 = self.pocket_list[index]
        rot = choice(range(24))
        tr = self.max_translation * np.random.rand(1, 3)
        x, y = self.feed_data(mol1, mol2, rot, tr)
        x = torch.Tensor(x.astype(np.float32)).permute(3, 0, 1, 2)
        y = torch.Tensor(y.astype(np.float32)).permute(3, 0, 1, 2)
        return x, y

    def gaussian_weight(self, distance, sigma):
        return np.exp(-(distance ** 2) / (2 * sigma ** 2))
    
    def make_grid_gaussian(self, coords, features, grid_resolution=1.0, max_dist=10.0, sigma=1.0):
        try:
            coords = np.asarray(coords, dtype=np.float64)
        except ValueError:
            raise ValueError('coords must be an array of floats of shape (N, 3)')
        if coords.shape[1] != 3:
            raise ValueError('coords must be of shape (N, 3)')
        try:
            features = np.asarray(features, dtype=np.float64)
        except ValueError:
            raise ValueError('features must be an array of floats of shape (N, F)')
        N, F = features.shape
        box_size = ceil(2 * max_dist / grid_resolution + 1)
        grid = np.zeros((box_size, box_size, box_size, F), dtype=np.float32)
        idx = np.arange(box_size)
        xv, yv, zv = np.meshgrid(idx, idx, idx, indexing='ij')
        grid_centers = np.stack([xv, yv, zv], axis=-1)
        grid_centers = grid_centers * grid_resolution - max_dist + grid_resolution/2
        for atom_coord, feat in zip(coords, features):
            atom_grid_coord = (atom_coord + max_dist) / grid_resolution
            window = int(np.ceil(3 * sigma / grid_resolution))
            lower_idx = np.maximum(np.floor(atom_grid_coord - window), 0).astype(int)
            upper_idx = np.minimum(np.floor(atom_grid_coord + window) + 1, box_size).astype(int)
            for i in range(lower_idx[0], upper_idx[0]):
                for j in range(lower_idx[1], upper_idx[1]):
                    for k in range(lower_idx[2], upper_idx[2]):
                        voxel_center = np.array([i, j, k]) * grid_resolution - max_dist + grid_resolution/2
                        distance = np.linalg.norm(voxel_center - atom_coord)
                        weight = self.gaussian_weight(distance, sigma)
                        grid[i, j, k, :] += feat * weight
        return grid

    def make_grid(self, coords, features, grid_resolution=1.0, max_dist=10.0):
        try:
            coords = np.asarray(coords, dtype=np.float64)
        except ValueError:
            raise ValueError('coords must be an array of floats of shape (N, 3)')
        if coords.shape[1] != 3:
            raise ValueError('coords must be of shape (N, 3)')
        try:
            features = np.asarray(features, dtype=np.float64)
        except ValueError:
            raise ValueError('features must be an array of floats of shape (N, F)')
        N, F = features.shape
        box_size = ceil(2 * max_dist / grid_resolution + 1)
        grid_coords = (coords + max_dist) / grid_resolution
        grid_coords = grid_coords.round().astype(int)
        in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
        grid = np.zeros((box_size, box_size, box_size, F), dtype=np.float32)
        for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
            grid[x, y, z] += f
        return grid

    def rotation_matrix(self, axis, theta):
        axis = np.asarray(axis, dtype=np.float64)
        if axis.shape != (3,):
            raise ValueError('axis must be an array of floats of shape (3,)')
        if not isinstance(theta, (float, int)):
            raise TypeError('theta must be a float')
        axis = axis / sqrt(np.dot(axis, axis))
        a = cos(theta / 2.0)
        b, c, d = -axis * sin(theta / 2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([[aa + bb - cc - dd, 2*(bc + ad), 2*(bd - ac)],
                         [2*(bc - ad), aa + cc - bb - dd, 2*(cd + ab)],
                         [2*(bd + ac), 2*(cd - ab), aa + dd - bb - cc]])

    def rotate(self, coords, rotation):
        ROTATIONS = [self.rotation_matrix([1,1,1], 0)]
        for a1 in range(3):
            for t in range(1, 4):
                axis = np.zeros(3)
                axis[a1] = 1
                theta = t * pi / 2.0
                ROTATIONS.append(self.rotation_matrix(axis, theta))
        for (a1, a2) in combinations(range(3), 2):
            axis = np.zeros(3)
            axis[[a1, a2]] = 1.0
            theta = pi
            ROTATIONS.append(self.rotation_matrix(axis, theta))
            axis[a2] = -1.0
            ROTATIONS.append(self.rotation_matrix(axis, theta))
        for t in [1, 2]:
            theta = t * 2 * pi / 3
            axis = np.ones(3)
            ROTATIONS.append(self.rotation_matrix(axis, theta))
            for a1 in range(3):
                axis = np.ones(3)
                axis[a1] = -1
                ROTATIONS.append(self.rotation_matrix(axis, theta))
        if isinstance(rotation, int):
            if rotation >= 0 and rotation < len(ROTATIONS):
                return np.dot(coords, ROTATIONS[rotation])
            else:
                raise ValueError('Invalid rotation number %s!' % rotation)
        elif isinstance(rotation, np.ndarray) and rotation.shape == (3, 3):
            return np.dot(coords, rotation)
        else:
            raise ValueError('Invalid rotation %s!' % rotation)

    def feed_data(self, mol1, mol2, rotation=0, translation=(0,0,0)):
        if not isinstance(mol1, pybel.Molecule):
            raise TypeError('mol should be a pybel.Molecule object, got %s instead' % type(mol1))
        if not isinstance(mol2, pybel.Molecule):
            raise TypeError('mol should be a pybel.Molecule object, got %s instead' % type(mol2))
        if self.featurizer is None:
            raise ValueError('featurizer must be set to make predistions for molecules')
        if self.scale is None:
            raise ValueError('scale must be set to make predistions')
        prot_coords, prot_features = self.featurizer.get_features(mol1)
        pocket_coords, pocket_features = self.featurizer.get_features_gt(mol2)
        centroid = prot_coords.mean(axis=0)
        prot_coords -= centroid
        prot_coords = self.rotate(prot_coords, rotation)
        prot_coords += translation
        resolution = 1. / self.scale

        if self.use_gaussian:
            x = self.make_grid_gaussian(prot_coords, prot_features,
                                        max_dist=self.max_dist,
                                        grid_resolution=resolution,
                                        sigma=self.sigma)
        else:
            x = self.make_grid(prot_coords, prot_features,
                               max_dist=self.max_dist,
                               grid_resolution=resolution)

        y_channels = pocket_features.shape[1]
        pocket_coords -= centroid
        pocket_coords = self.rotate(pocket_coords, rotation)
        pocket_coords += translation

        gt = self.make_grid(pocket_coords, pocket_features,
                            max_dist=self.max_dist)
        margin = ndimage.maximum_filter(gt, footprint=self.footprint)
        gt += margin
        gt = gt.clip(0, 1)
        zoom = x.shape[1] / gt.shape[1]
        gt = np.expand_dims(gt, 0)
        gt = np.stack([ndimage.zoom(gt[0, ..., i], zoom) for i in range(y_channels)], -1)
        gt = gt.clip(0, 1)
        return x, gt


class proteinDataset_predict(Dataset):
    def __init__(self, data_path, featurizer=Featurizer(save_molecule_codes=False),
                 max_dist=35, scale=0.5, max_translation=5,
                 file_format='mol2',
                 use_gaussian=True, sigma=1.0):
        self.data_path = data_path
        self.max_dist = max_dist
        self.scale = scale
        self.max_translation = max_translation
        self.featurizer = featurizer
        self.file_format = file_format
        self.use_gaussian = use_gaussian
        self.sigma = sigma
        self.data_list = [os.path.join(self.data_path, x) for x in sorted(os.listdir(self.data_path))]
        footprint = ellipsoid(2, 2, 2)
        self.footprint = footprint.reshape((*footprint.shape, 1))
        self.protein_list = [next(pybel.readfile(self.file_format, os.path.join(path, 'protein.' + self.file_format)))
                              for path in self.data_list]

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        molname = self.data_list[index].split('/')[-1]
        mol_path = os.path.join(self.data_list[index], 'protein.' + self.file_format)
        mol1 = self.protein_list[index]
        x, origin, step = self.feed_data(mol1)
        x = torch.Tensor(x.astype(np.float32)).permute(3, 0, 1, 2)
        return x, origin, step, molname, mol_path

    def gaussian_weight(self, distance, sigma):
        return np.exp(-(distance ** 2) / (2 * sigma ** 2))
    
    def make_grid_gaussian(self, coords, features, grid_resolution=1.0, max_dist=10.0, sigma=1.0):
        try:
            coords = np.asarray(coords, dtype=np.float64)
        except ValueError:
            raise ValueError('coords must be an array of floats of shape (N, 3)')
        if coords.shape[1] != 3:
            raise ValueError('coords must be of shape (N, 3)')
        try:
            features = np.asarray(features, dtype=np.float64)
        except ValueError:
            raise ValueError('features must be an array of floats of shape (N, F)')
        N, F = features.shape
        box_size = ceil(2 * max_dist / grid_resolution + 1)
        grid = np.zeros((box_size, box_size, box_size, F), dtype=np.float32)
        idx = np.arange(box_size)
        xv, yv, zv = np.meshgrid(idx, idx, idx, indexing='ij')
        grid_centers = np.stack([xv, yv, zv], axis=-1)
        grid_centers = grid_centers * grid_resolution - max_dist + grid_resolution/2
        for atom_coord, feat in zip(coords, features):
            atom_grid_coord = (atom_coord + max_dist) / grid_resolution
            window = int(np.ceil(3 * sigma / grid_resolution))
            lower_idx = np.maximum(np.floor(atom_grid_coord - window), 0).astype(int)
            upper_idx = np.minimum(np.floor(atom_grid_coord + window) + 1, box_size).astype(int)
            for i in range(lower_idx[0], upper_idx[0]):
                for j in range(lower_idx[1], upper_idx[1]):
                    for k in range(lower_idx[2], upper_idx[2]):
                        voxel_center = np.array([i, j, k]) * grid_resolution - max_dist + grid_resolution/2
                        distance = np.linalg.norm(voxel_center - atom_coord)
                        weight = self.gaussian_weight(distance, sigma)
                        grid[i, j, k, :] += feat * weight
        return grid

    def make_grid(self, coords, features, grid_resolution=1.0, max_dist=10.0):
        try:
            coords = np.asarray(coords, dtype=np.float64)
        except ValueError:
            raise ValueError('coords must be an array of floats of shape (N, 3)')
        if coords.shape[1] != 3:
            raise ValueError('coords must be of shape (N, 3)')
        try:
            features = np.asarray(features, dtype=np.float64)
        except ValueError:
            raise ValueError('features must be an array of floats of shape (N, F)')
        N, F = features.shape
        box_size = ceil(2 * max_dist / grid_resolution + 1)
        grid_coords = (coords + max_dist) / grid_resolution
        grid_coords = grid_coords.round().astype(int)
        in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(axis=1)
        grid = np.zeros((box_size, box_size, box_size, F), dtype=np.float32)
        for (x, y, z), f in zip(grid_coords[in_box], features[in_box]):
            grid[x, y, z] += f
        return grid

    def feed_data(self, mol1):
        if not isinstance(mol1, pybel.Molecule):
            raise TypeError('mol should be a pybel.Molecule object, got %s instead' % type(mol1))
        if self.featurizer is None:
            raise ValueError('featurizer must be set to make predistions')
        if self.scale is None:
            raise ValueError('scale must be set to make predistions')
        prot_coords, prot_features = self.featurizer.get_features(mol1)
        centroid = prot_coords.mean(axis=0)
        prot_coords -= centroid
        resolution = 1. / self.scale
        if self.use_gaussian:
            x = self.make_grid_gaussian(prot_coords, prot_features,
                                        max_dist=self.max_dist,
                                        grid_resolution=resolution,
                                        sigma=self.sigma)
        else:
            x = self.make_grid(prot_coords, prot_features,
                               max_dist=self.max_dist,
                               grid_resolution=resolution)
        origin = centroid - self.max_dist
        step = np.array([1.0 / self.scale] * 3)
        return x, origin, step

