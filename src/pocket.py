import numpy as np
from rdkit import Chem  # pylint: disable=no-member
from rdkit.Geometry import Point3D  # pylint: disable=no-member
from scipy.ndimage import binary_erosion, label

def build_pocket_grid(protein, ref_ligand, padding=4.0, spacing=0.5, probe_radius=1.4,
                      vdw_radii=None):
    """Voxelise the region around the reference ligand and mark empty voxels.

    Returns (origin, spacing, empty_bool_array, ref_centroid).
    """
    ref_conf = ref_ligand.GetConformer()
    ref_coords = np.array([
        list(ref_conf.GetAtomPosition(i))
        for i in range(ref_ligand.GetNumAtoms())
        if ref_ligand.GetAtomWithIdx(i).GetAtomicNum() > 1
    ])
    ref_centroid = ref_coords.mean(axis=0)

    lo = ref_coords.min(axis=0) - padding
    hi = ref_coords.max(axis=0) + padding
    dims = np.ceil((hi - lo) / spacing).astype(int)

    xs = lo[0] + (np.arange(dims[0]) + 0.5) * spacing
    ys = lo[1] + (np.arange(dims[1]) + 0.5) * spacing
    zs = lo[2] + (np.arange(dims[2]) + 0.5) * spacing
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1)

    if vdw_radii is None:
        vdw_radii = {}

    occupied = np.zeros(dims, dtype=bool)
    prot_conf = protein.GetConformer()
    prot_coords = np.array([
        list(prot_conf.GetAtomPosition(i)) for i in range(protein.GetNumAtoms())
    ])
    box_centre = (lo + hi) / 2
    box_radius = np.linalg.norm(hi - lo) / 2

    for atom in protein.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue
        i = atom.GetIdx()
        if np.linalg.norm(prot_coords[i] - box_centre) > box_radius + 5.0:
            continue
        r = vdw_radii.get(atom.GetAtomicNum(), 1.7) + probe_radius
        d2 = np.sum((grid - prot_coords[i]) ** 2, axis=-1)
        occupied |= d2 < r * r

    return lo, spacing, ~occupied, ref_centroid


def pocket_to_pseudomol(empty, origin, spacing, ref_centroid,
                        surface_only=True, subsample=2, dummy_element=2):
    """Convert empty-voxel grid to a bondless RDKit Mol of dummy atoms.

    Only the connected empty component containing the reference centroid
    is kept. Returns (mol, pocket_mask).
    """
    labels, _ = label(empty)
    idx = np.floor((ref_centroid - origin) / spacing).astype(int)
    if not np.all((idx >= 0) & (idx < np.array(empty.shape))):
        raise ValueError("Reference centroid is outside the grid — increase padding.")
    target_label = labels[tuple(idx)]
    if target_label == 0:
        raise ValueError(
            "Reference centroid landed on an occupied voxel. "
            "Try a smaller probe_radius or finer spacing."
        )
    pocket_mask = labels == target_label

    pocket = pocket_mask.copy()
    if surface_only:
        pocket = pocket & ~binary_erosion(pocket)
    if subsample > 1:
        mask = np.zeros_like(pocket)
        mask[::subsample, ::subsample, ::subsample] = True
        pocket = pocket & mask

    ii, jj, kk = np.where(pocket)
    coords = origin + (np.stack([ii, jj, kk], axis=1) + 0.5) * spacing

    rw = Chem.RWMol()
    for _ in range(len(coords)):
        rw.AddAtom(Chem.Atom(dummy_element))
    conf = Chem.Conformer(len(coords))
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    rw.AddConformer(conf, assignId=True)
    return rw.GetMol(), pocket_mask
