# SwinSite: Protein–Ligand Binding Site Prediction

**SwinSite** is a deep learning-based method for predicting protein–ligand binding sites using a hybrid of Swin Transformer and 3D CNN architectures. It processes protein structures in 3D and outputs likely binding regions.

---

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yaml
conda activate swinsite
```

---

## Usage

### Running the Prediction

Run the prediction script on a set of protein–ligand complexes:

```bash
python predict.py \
    -i ./example \
    -f mol2 \
    -o ./output/example \
    -l ./logs/log.txt
```

**Arguments**:

- `-i` : Path to input directory (should contain subdirectories with `protein.pdb` and `ligand.mol2`)
- `-f` : File format (`mol2` or `pdb`)
- `-o` : Output directory
- `-l` : Log file path

The predicted pockets and grids will be saved in the specified output directory.

---

### Jupyter Notebook Example

To run both prediction and visualization in an interactive environment:

1. Launch Jupyter Lab:

    ```bash
    jupyter lab
    ```

2. Open the notebook:

    ```
    predict_and_visualize.ipynb
    ```

This notebook:
- Runs `predict.py` for inference
- Uses `py3Dmol` to visualize proteins, ligands, and predicted pockets

---

## Example Folder Structure

```
example/
├── 1abc/
│   ├── protein.pdb
│   ├── ligand.mol2
├── 2xyz/
│   ├── protein.pdb
│   ├── ligand.mol2
```

Each subdirectory should contain a single protein–ligand pair.

---

## Output Files

For each processed sample, the following files will be generated:

- `pocket_*.mol2` : Atom-level predicted pockets
- `grid_*.mol2`   : Grid-based volumetric pockets
- `log.txt`       : Log file containing progress and error messages

---
