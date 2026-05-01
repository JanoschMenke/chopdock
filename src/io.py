from rdkit import Chem  # pylint: disable=no-member


def load_reference_ligand(path):
    suffix = path.lower().rsplit(".", 1)[-1]
    if suffix == "sdf":
        suppl = Chem.SDMolSupplier(path, removeHs=False)
        mol = next((m for m in suppl if m is not None), None)
    elif suffix == "mol":
        mol = Chem.MolFromMolFile(path, removeHs=False)
    elif suffix == "pdb":
        mol = Chem.MolFromPDBFile(path, removeHs=False)
    else:
        raise ValueError(f"Unsupported reference ligand format: .{suffix}")
    if mol is None:
        raise ValueError(f"Could not read reference ligand: {path}")
    return mol


def write_sdf(mol, path, conf_id=-1, **props):
    for k, v in props.items():
        mol.SetProp(k, str(v))
    w = Chem.SDWriter(path)
    w.write(mol, confId=conf_id)
    w.close()
