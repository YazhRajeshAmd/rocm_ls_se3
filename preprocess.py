import os
import torch
import numpy as np
import dgl
from rdkit import Chem
from Bio.PDB import PDBParser
from tqdm import tqdm

DATA_DIR = "./pbpp-2020"
OUTPUT_DIR = "./processed"
CUTOFF = 5.0  # distance cutoff in Å

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_ligand_atoms(mol2_path):
    mol = Chem.MolFromMol2File(mol2_path)
    conf = mol.GetConformer()
    coords = []
    feats = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        feats.append([atom.GetAtomicNum()])  # simple feature

    return np.array(coords), np.array(feats)


def get_protein_atoms(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    coords = []
    feats = []

    for atom in structure.get_atoms():
        coords.append(atom.coord)
        feats.append([atom.element])  # will encode later

    return np.array(coords), feats


def encode_protein_elements(feats):
    element_map = {"C":6, "N":7, "O":8, "S":16}
    encoded = []
    for f in feats:
        encoded.append([element_map.get(f[0], 0)])
    return np.array(encoded)


def build_graph(coords):
    num_atoms = len(coords)
    src, dst = [], []

    for i in range(num_atoms):
        for j in range(num_atoms):
            if i != j:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < CUTOFF:
                    src.append(i)
                    dst.append(j)

    g = dgl.graph((src, dst), num_nodes=num_atoms)
    return g


def load_affinity(index_file):
    affinity_dict = {}
    with open(index_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            pdb_id = parts[0]
            affinity = float(parts[3])
            affinity_dict[pdb_id] = affinity
    return affinity_dict


def main():
    index_path = os.path.join(DATA_DIR, "INDEX_refined_data.2020")
    affinity_dict = load_affinity(index_path)

    for pdb_id in tqdm(os.listdir(DATA_DIR)):
        complex_dir = os.path.join(DATA_DIR, pdb_id)
        if not os.path.isdir(complex_dir):
            continue

        ligand_path = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")
        protein_path = os.path.join(complex_dir, f"{pdb_id}_pocket.pdb")

        if not os.path.exists(ligand_path):
            continue

        lig_coords, lig_feats = get_ligand_atoms(ligand_path)
        prot_coords, prot_feats = get_protein_atoms(protein_path)
        prot_feats = encode_protein_elements(prot_feats)

        coords = np.vstack([lig_coords, prot_coords])
        feats = np.vstack([lig_feats, prot_feats])

        graph = build_graph(coords)

        sample = {
            "graph": graph,
            "node_feats": torch.tensor(feats).float(),
            "coords": torch.tensor(coords).float(),
            "label": affinity_dict.get(pdb_id, 0.0)
        }

        torch.save(sample, os.path.join(OUTPUT_DIR, f"{pdb_id}.pt"))


if __name__ == "__main__":
    main()
