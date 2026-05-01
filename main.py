from rdkit import Chem  # pylint: disable=no-member

from src.alignment import align_to_reference, dock_candidate, mmff_energy, score_pose, set_coords
from src.config import Config
from src.conformers import embed_candidate
from src.io import load_reference_ligand, write_sdf
from src.pocket import build_pocket_grid, pocket_to_pseudomol
from src.refinement import minimize_pose, refine_pose


def main(cfg: Config = Config()):
    print("Loading structures…")
    protein = Chem.MolFromPDBFile(cfg.protein_pdb, sanitize=False, removeHs=False,
                                  proximityBonding=False)
    if protein is None:
        raise ValueError(f"Could not read protein: {cfg.protein_pdb}")
    reference = load_reference_ligand(cfg.reference_ligand)

    print("Building pocket grid…")
    origin, spacing, empty, ref_centroid = build_pocket_grid(
        protein, reference,
        padding=cfg.padding, spacing=cfg.spacing, probe_radius=cfg.probe_radius,
        vdw_radii=cfg.vdw_radii,
    )
    print(f"  grid dims: {empty.shape}, total empty voxels: {empty.sum()}")

    print("Building pseudo-pocket molecule…")
    pseudo, pocket_mask = pocket_to_pseudomol(
        empty, origin, spacing, ref_centroid,
        surface_only=cfg.surface_only, subsample=cfg.subsample,
    )
    print(f"  pseudo-pocket atoms: {pseudo.GetNumAtoms()}")
    if cfg.out_pocket_sdf:
        write_sdf(pseudo, cfg.out_pocket_sdf)
        print(f"  wrote pocket cloud to {cfg.out_pocket_sdf}")

    print(f"Embedding candidate ({cfg.candidate_smiles})…")
    candidate = embed_candidate(cfg.candidate_smiles, n_confs=cfg.n_confs)
    print(f"  generated {candidate.GetNumConformers()} conformers")

    ref_score = score_pose(reference, 0, pseudo)
    print(f"  reference ligand ProtrudeDist: {ref_score:.4f} (baseline)")

    print("O3A alignment…")
    o3a_best = float("inf")
    for cid in range(candidate.GetNumConformers()):
        try:
            align_to_reference(candidate, reference, cid)
        except Exception:  # noqa: BLE001
            continue
        s = score_pose(candidate, cid, pseudo)
        if s < o3a_best:
            o3a_best = s
    print(f"  best O3A-only ProtrudeDist: {o3a_best:.4f}")

    print("Docking…")
    best_cid, best_score, best_coords = dock_candidate(
        candidate, reference, pseudo, n_jitters=cfg.n_jitters,
    )
    print(f"  best conformer: {best_cid}, ProtrudeDist = {best_score:.4f}")
    print("  (0 = perfectly contained, 1 = entirely outside pocket)")

    set_coords(candidate, best_cid, best_coords)

    if cfg.refine_pose:
        print("Refining pose…")

        if cfg.out_original_sdf:
            write_sdf(candidate, cfg.out_original_sdf, conf_id=best_cid,
                      ProtrudeDist=f"{best_score:.4f}")
            print(f"  wrote pre-refinement pose to {cfg.out_original_sdf}")

        e_before = mmff_energy(candidate, best_cid)
        best_score = refine_pose(
            candidate, best_cid, pseudo, pocket_mask, origin, spacing,
            step_deg=cfg.refine_step_deg, max_iter=cfg.refine_max_iter,
        )
        e_after = mmff_energy(candidate, best_cid)
        print(f"  ProtrudeDist after torsion refinement: {best_score:.4f}")
        if e_before is not None and e_after is not None:
            print(f"  MMFF strain introduced: {e_after - e_before:+.2f} kcal/mol")

        if cfg.minimize_after_refine:
            e_post_min = minimize_pose(
                candidate, best_cid, pocket_mask, origin, spacing,
                core_force=cfg.minimize_core_force,
                max_displacement=cfg.minimize_max_displacement,
            )
            best_score = score_pose(candidate, best_cid, pseudo)
            print(f"  ProtrudeDist after minimization: {best_score:.4f}", end="")
            if e_after is not None and e_post_min is not None:
                print(f", MMFF strain relieved: {e_after - e_post_min:+.2f} kcal/mol", end="")
            print()

    write_sdf(candidate, cfg.out_pose_sdf, conf_id=best_cid,
              ProtrudeDist=f"{best_score:.4f}")
    print(f"Wrote best pose to {cfg.out_pose_sdf}")


if __name__ == "__main__":
    main()
