import numpy as np
from rdkit import rdBase  # pylint: disable=no-member
from rdkit.Chem import AllChem, rdMolAlign, rdShapeHelpers  # pylint: disable=no-member
from rdkit.Geometry import Point3D  # pylint: disable=no-member
from scipy.spatial.transform import Rotation


def score_pose(mol, conf_id, pseudo_pocket):
    """Lower = better. Fraction of candidate volume sticking out of the pocket."""
    return rdShapeHelpers.ShapeProtrudeDist(
        mol, pseudo_pocket, confId1=conf_id, confId2=0, allowReordering=False
    )


def mmff_energy(mol, conf_id):
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
    return ff.CalcEnergy() if ff is not None else None


def align_to_reference(candidate, reference, conf_id):
    with rdBase.BlockLogs():
        cand_props = AllChem.MMFFGetMoleculeProperties(candidate)
        ref_props = AllChem.MMFFGetMoleculeProperties(reference)
        o3a = rdMolAlign.GetO3A(
            candidate, reference, cand_props, ref_props, prbCid=conf_id, refCid=0
        )
        o3a.Align()
    return o3a.Score()


def get_coords(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])


def set_coords(mol, conf_id, coords):
    conf = mol.GetConformer(conf_id)
    for i, xyz in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def jitter_conformer(mol, conf_id, rotation, translation):
    """Apply a rigid-body transform around the conformer centroid.
    Returns original coords so the caller can restore them."""
    coords = get_coords(mol, conf_id)
    centroid = coords.mean(axis=0)
    set_coords(mol, conf_id, rotation.apply(coords - centroid) + centroid + translation)
    return coords


def dock_candidate(candidate, reference, pseudo_pocket,
                   n_jitters=20, jitter_angle_deg=15.0, jitter_translation=1.0,
                   seed=0xC0DE):
    rng = np.random.default_rng(seed)
    best = (None, np.inf, None)  # (conf_id, score, coords)

    for cid in range(candidate.GetNumConformers()):
        try:
            align_to_reference(candidate, reference, cid)
        except Exception as e:  # noqa: BLE001
            print(f"  conf {cid}: O3A failed ({e}), skipping")
            continue

        score = score_pose(candidate, cid, pseudo_pocket)
        coords = get_coords(candidate, cid)
        if score < best[1]:
            best = (cid, score, coords.copy())

        for _ in range(n_jitters):
            angle = rng.uniform(-jitter_angle_deg, jitter_angle_deg, size=3)
            rot = Rotation.from_euler("xyz", angle, degrees=True)
            trans = rng.uniform(-jitter_translation, jitter_translation, size=3)
            saved = jitter_conformer(candidate, cid, rot, trans)
            try:
                s = score_pose(candidate, cid, pseudo_pocket)
                if s < best[1]:
                    best = (cid, s, get_coords(candidate, cid).copy())
            finally:
                set_coords(candidate, cid, saved)

    if best[0] is None:
        raise RuntimeError("No pose could be scored.")
    return best
