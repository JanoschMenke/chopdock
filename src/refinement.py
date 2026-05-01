import numpy as np
from rdkit.Chem import AllChem  # pylint: disable=no-member
from rdkit.Geometry import Point3D  # pylint: disable=no-member
from scipy.spatial.transform import Rotation

from .alignment import get_coords, set_coords, score_pose


def get_rotatable_bonds(mol):
    return [
        bond for bond in mol.GetBonds()
        if bond.GetBondTypeAsDouble() == 1.0
        and not bond.IsInRing()
        and bond.GetBeginAtom().GetDegree() > 1
        and bond.GetEndAtom().GetDegree() > 1
    ]


def atoms_on_side(mol, bond, start_idx):
    other = bond.GetOtherAtomIdx(start_idx)
    visited, stack = set(), [start_idx]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for nb in mol.GetAtomWithIdx(cur).GetNeighbors():
            if nb.GetIdx() != other:
                stack.append(nb.GetIdx())
    return visited


def get_protruding_atoms(mol, conf_id, pocket_mask, origin, spacing):
    """Heavy atom indices outside the pocket — in protein or bulk solvent."""
    conf = mol.GetConformer(conf_id)
    dims = np.array(pocket_mask.shape)
    protruding = set()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
        idx = np.floor((pos - origin) / spacing).astype(int)
        if not np.all((idx >= 0) & (idx < dims)) or not pocket_mask[tuple(idx)]:
            protruding.add(atom.GetIdx())
    return protruding


def rotate_around_bond(mol, conf_id, anchor_idx, pivot_idx, angle_deg, moving_atoms):
    conf = mol.GetConformer(conf_id)
    pivot = np.array(conf.GetAtomPosition(pivot_idx))
    axis = pivot - np.array(conf.GetAtomPosition(anchor_idx))
    axis /= np.linalg.norm(axis)
    rot = Rotation.from_rotvec(np.radians(angle_deg) * axis)
    for idx in moving_atoms:
        pos = np.array(conf.GetAtomPosition(idx)) - pivot
        new_pos = rot.apply(pos) + pivot
        conf.SetAtomPosition(idx, Point3D(float(new_pos[0]), float(new_pos[1]), float(new_pos[2])))


def refine_pose(mol, conf_id, pseudo_pocket, pocket_mask, origin, spacing,
                step_deg=10, max_iter=3):
    rot_bonds = get_rotatable_bonds(mol)
    current_score = score_pose(mol, conf_id, pseudo_pocket)

    for _ in range(max_iter):
        improved = False
        protruding = get_protruding_atoms(mol, conf_id, pocket_mask, origin, spacing)
        if not protruding:
            break

        for bond in rot_bonds:
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            side_begin = atoms_on_side(mol, bond, begin)
            side_end = atoms_on_side(mol, bond, end)

            if not (protruding & side_begin) and not (protruding & side_end):
                continue

            if len(side_begin) <= len(side_end):
                moving, pivot, anchor = side_begin, end, begin
            else:
                moving, pivot, anchor = side_end, begin, end

            conf = mol.GetConformer(conf_id)
            saved = {i: np.array(conf.GetAtomPosition(i)) for i in moving}

            best_angle, best_bond_score = 0.0, current_score
            for angle in np.arange(step_deg, 360, step_deg):
                for i, pos in saved.items():
                    conf.SetAtomPosition(i, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
                rotate_around_bond(mol, conf_id, anchor, pivot, angle, moving)
                s = score_pose(mol, conf_id, pseudo_pocket)
                if s < best_bond_score:
                    best_bond_score, best_angle = s, angle

            for i, pos in saved.items():
                conf.SetAtomPosition(i, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
            if best_angle > 0.0:
                rotate_around_bond(mol, conf_id, anchor, pivot, best_angle, moving)
                current_score = best_bond_score
                improved = True

        if not improved:
            break

    return current_score


def minimize_pose(mol, conf_id, pocket_mask, origin, spacing,
                  core_force=500.0, max_displacement=0.5):
    """Constrained MMFF minimization: freeze pocket-buried atoms, relax the rest."""
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    if ff is None:
        return None

    conf = mol.GetConformer(conf_id)
    dims = np.array(pocket_mask.shape)
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = np.array(conf.GetAtomPosition(atom.GetIdx()))
        idx = np.floor((pos - origin) / spacing).astype(int)
        if np.all((idx >= 0) & (idx < dims)) and pocket_mask[tuple(idx)]:
            ff.MMFFAddPositionConstraint(atom.GetIdx(), max_displacement, core_force)

    ff.Minimize()
    return ff.CalcEnergy()
