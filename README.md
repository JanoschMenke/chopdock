# ChopDock

You want to align your molecule to a reference ligand, but there's always this pesky protein in the way.

You could install Smina, wrestle with dependencies, prepare the files, and three hours later have a pose. But I just wanted to jam my molecule into this protein pocket.

ChopDock voxelises the binding pocket from a reference ligand and protein PDB, converts it into a pseudo-molecule, and optimises how well your candidate fits inside using RDKit's shape overlap scoring. A bit of O3A alignment to get started, some random jitter to escape fit the pocket, a torsion sweep to tuck in any sticking-out bits, done.

This is not real docking. There is no scoring function, no electrostatics and most importantly no repulsion. So there might be clashes. But in practice, the poses look noticeably more reasonably aligned than plain O3A, which ignores the protein entirely. It's a decent starting point without any of the setup pain.

The only hard dependencies are RDKit and SciPy.

## Usage
Try the walkthrough in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JanoschMenke/chopdock/blob/main/walkthrough.ipynb)
```bash
# install
uv sync

# run with default config
uv run python main.py

# step-by-step walkthrough notebook
uv run jupyter notebook walkthrough.ipynb
```

To change inputs or parameters, edit the `Config` dataclass in `src/config.py`.
