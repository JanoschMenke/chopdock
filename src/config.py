from dataclasses import dataclass, field


@dataclass
class Config:
    protein_pdb: str = "data/example_data/7XKJ_protein.pdb"
    reference_ligand: str = "data/example_data/7XKJ_ligand.sdf"
    candidate_smiles: str = "CC1C=CC(NC(C2C=CC(C3C=CC=C3)=CC=2)=O)=CC=1NC1N=C(C2C=CC=NC=2)C=CN=1"
    out_pose_sdf: str = "data/results/best_pose.sdf"
    out_original_sdf: str | None = "data/results/best_pose_before_refinment.sdf"
    out_pocket_sdf: str | None = "data/results/pocket_cloud.sdf"

    n_confs: int = 100
    n_jitters: int = 20
    padding: float = 4.0
    spacing: float = 0.5
    probe_radius: float = 1.4
    subsample: int = 2
    surface_only: bool = True
    refine_pose: bool = True
    refine_step_deg: int = 5
    refine_max_iter: int = 3
    minimize_after_refine: bool = True
    minimize_core_force: float = 500.0
    minimize_max_displacement: float = 0.5

    vdw_radii: dict[int, float] = field(default_factory=lambda: {
        1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47,
        11: 2.27, 12: 1.73, 15: 1.80, 16: 1.80, 17: 1.75,
        19: 2.75, 20: 2.31, 26: 2.00, 30: 1.39, 35: 1.85, 53: 1.98,
    })
