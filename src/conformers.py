from rdkit import Chem  # pylint: disable=no-member
from rdkit.Chem import AllChem  # pylint: disable=no-member


def embed_candidate(smiles, n_confs=100, seed=0xF00D):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.5
    if not AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params):
        raise RuntimeError("Embedding failed for all conformers.")
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    return mol
