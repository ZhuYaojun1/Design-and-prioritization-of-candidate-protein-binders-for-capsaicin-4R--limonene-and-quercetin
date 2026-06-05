from pathlib import Path
import csv
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

# ============================================================
# Paths
# ============================================================

ROOT = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\6_AutoDock_Vina\9IR")
FORMAL = ROOT / "Docking_48_Formal"
QC_DIR = FORMAL / "QC"

TEMPLATE_SDF = ROOT / "9IR_PDBQT" / "9IR_ideal.sdf"
REFERENCE_DIR = ROOT / "box_parameters" / "reference_ligands"
DOCKING_SDF_DIR = FORMAL / "outputs_sdf"

PREVIOUS_METRICS = QC_DIR / "mode1_spatial_overlap_metrics_9IR.csv"

OUTPUT_METRICS = QC_DIR / "mode1_topology_aware_in_place_rmsd_metrics_9IR.csv"
OUTPUT_RANKING = QC_DIR / "mode1_topology_aware_MD_priority_ranking_9IR.csv"

# ============================================================
# Read molecules
# ============================================================

def read_first_sdf_molecule(path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    raise RuntimeError(f"Unable to read molecule from: {path}")

def heavy_atom_molecule(mol):
    out = Chem.RemoveHs(mol)
    Chem.SanitizeMol(out)
    return out

template = heavy_atom_molecule(read_first_sdf_molecule(TEMPLATE_SDF))
template_smiles = Chem.MolToSmiles(template)

def build_reference_molecule(reference_pdb):
    """
    Read the RFD3 reference 9IR coordinates from PDB, then assign the
    chemically correct 9IR bond topology and bond orders from 9IR_ideal.sdf.
    Coordinates remain those of the RFD3 reference pose.
    """
    raw = Chem.MolFromPDBFile(
        str(reference_pdb),
        sanitize=False,
        removeHs=False,
        proximityBonding=True
    )

    if raw is None:
        raise RuntimeError(f"Unable to read reference PDB: {reference_pdb}")

    raw = Chem.RemoveHs(raw, sanitize=False)

    try:
        reference = AllChem.AssignBondOrdersFromTemplate(template, raw)
        Chem.SanitizeMol(reference)
    except Exception as exc:
        raise RuntimeError(
            f"Reference topology assignment failed for {reference_pdb.name}: {exc}"
        )

    reference_smiles = Chem.MolToSmiles(reference)

    if reference_smiles != template_smiles:
        raise RuntimeError(
            f"Reference topology mismatch: {reference_smiles} != {template_smiles}"
        )

    return reference

def read_docked_mode1_molecule(docked_sdf):
    """
    The first molecule in the Meeko-exported SDF corresponds to Vina mode 1.
    """
    docked = heavy_atom_molecule(read_first_sdf_molecule(docked_sdf))
    docked_smiles = Chem.MolToSmiles(docked)

    if docked_smiles != template_smiles:
        raise RuntimeError(
            f"Docked topology mismatch: {docked_smiles} != {template_smiles}"
        )

    return docked

# ============================================================
# Read previous contact metrics
# ============================================================

with open(PREVIOUS_METRICS, "r", encoding="utf-8-sig", newline="") as handle:
    previous_rows = list(csv.DictReader(handle))

successful_previous = [
    row for row in previous_rows
    if row.get("analysis_status") == "success"
]

print(f"Previous successful metric records found: {len(successful_previous)}")

if len(successful_previous) != 48:
    raise RuntimeError(
        f"Expected 48 successful previous metric rows, found {len(successful_previous)}"
    )

results = []

# ============================================================
# Calculate topology-aware in-place RMSD
# ============================================================

for row in successful_previous:

    receptor_id = row["receptor_id"]

    reference_pdb = REFERENCE_DIR / f"{receptor_id}_9IR_reference.pdb"
    docked_sdf = DOCKING_SDF_DIR / f"{receptor_id}_9IR_out.sdf"

    try:
        reference = build_reference_molecule(reference_pdb)
        docked = read_docked_mode1_molecule(docked_sdf)

        # RDKit CalcRMS:
        # - does not align or move docked ligand before RMSD calculation;
        # - considers valid symmetry/topology-based atom mappings.
        topology_rmsd = rdMolAlign.CalcRMS(
            docked,
            reference,
            maxMatches=100000
        )

        affinity = float(row["vina_affinity_kcal_mol"])
        pocket_rmsd = float(row["pocket_alignment_rmsd_A"])
        centroid_shift = float(row["centroid_shift_A"])
        recovery = float(row["contact_recovery_percent"])
        new_contacts = float(row["new_contact_percent_of_docked"])
        jaccard = float(row["contact_jaccard"])
        pocket_risk = row["pocket_alignment_risk"]

        # ====================================================
        # Final redocking tier using topology-aware in-place RMSD
        # ====================================================

        if (
            topology_rmsd <= 2.0
            and centroid_shift <= 1.0
            and recovery >= 80.0
            and new_contacts <= 20.0
            and pocket_risk == "Acceptable_pocket_alignment"
        ):
            tier = "Tier_A_topology_supported_pose"

        elif (
            topology_rmsd <= 3.0
            and centroid_shift <= 2.0
            and recovery >= 60.0
            and new_contacts <= 35.0
        ):
            tier = "Tier_B_partially_supported_pose"

        else:
            tier = "Tier_C_low_priority_pose"

        # ====================================================
        # MD selection bucket
        # ====================================================

        if tier == "Tier_A_topology_supported_pose" and affinity <= -6.8:
            md_bucket = "Primary_MD_shortlist"

        elif tier == "Tier_A_topology_supported_pose":
            md_bucket = "Pose_supported_backup"

        elif tier == "Tier_B_partially_supported_pose" and affinity <= -7.0:
            md_bucket = "Strong_affinity_manual_review"

        else:
            md_bucket = "Not_priority_for_initial_MD"

        results.append({
            "receptor_id": receptor_id,
            "analysis_status": "success",
            "vina_affinity_kcal_mol": round(affinity, 4),
            "topology_aware_in_place_rmsd_A": round(float(topology_rmsd), 4),
            "centroid_shift_A": round(centroid_shift, 4),
            "contact_recovery_percent": round(recovery, 4),
            "new_contact_percent_of_docked": round(new_contacts, 4),
            "contact_jaccard": round(jaccard, 4),
            "pocket_alignment_rmsd_A": round(pocket_rmsd, 4),
            "pocket_alignment_risk": pocket_risk,
            "redocking_validation_tier": tier,
            "md_selection_bucket": md_bucket,
            "reference_contact_residues": row.get("reference_contact_residues", ""),
            "docked_contact_residues": row.get("docked_contact_residues", ""),
            "recovered_contact_residues": row.get("recovered_contact_residues", ""),
            "new_contact_residues": row.get("new_contact_residues", "")
        })

    except Exception as exc:
        results.append({
            "receptor_id": receptor_id,
            "analysis_status": "analysis_failed",
            "error": str(exc)
        })

# ============================================================
# Save complete metrics
# ============================================================

metric_fields = [
    "receptor_id",
    "analysis_status",
    "vina_affinity_kcal_mol",
    "topology_aware_in_place_rmsd_A",
    "centroid_shift_A",
    "contact_recovery_percent",
    "new_contact_percent_of_docked",
    "contact_jaccard",
    "pocket_alignment_rmsd_A",
    "pocket_alignment_risk",
    "redocking_validation_tier",
    "md_selection_bucket",
    "reference_contact_residues",
    "docked_contact_residues",
    "recovered_contact_residues",
    "new_contact_residues",
    "error"
]

with open(OUTPUT_METRICS, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=metric_fields)
    writer.writeheader()

    for result in results:
        writer.writerow({key: result.get(key, "") for key in metric_fields})

# ============================================================
# Ranking
# ============================================================

successful = [
    row for row in results
    if row.get("analysis_status") == "success"
]

bucket_order = {
    "Primary_MD_shortlist": 0,
    "Pose_supported_backup": 1,
    "Strong_affinity_manual_review": 2,
    "Not_priority_for_initial_MD": 3
}

tier_order = {
    "Tier_A_topology_supported_pose": 0,
    "Tier_B_partially_supported_pose": 1,
    "Tier_C_low_priority_pose": 2
}

ranked = sorted(
    successful,
    key=lambda row: (
        bucket_order.get(row["md_selection_bucket"], 9),
        tier_order.get(row["redocking_validation_tier"], 9),
        float(row["vina_affinity_kcal_mol"]),
        float(row["topology_aware_in_place_rmsd_A"]),
        -float(row["contact_recovery_percent"]),
        float(row["new_contact_percent_of_docked"])
    )
)

ranking_fields = [
    "receptor_id",
    "md_selection_bucket",
    "redocking_validation_tier",
    "vina_affinity_kcal_mol",
    "topology_aware_in_place_rmsd_A",
    "centroid_shift_A",
    "contact_recovery_percent",
    "new_contact_percent_of_docked",
    "contact_jaccard",
    "pocket_alignment_rmsd_A",
    "pocket_alignment_risk"
]

with open(OUTPUT_RANKING, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=ranking_fields)
    writer.writeheader()

    for result in ranked:
        writer.writerow({key: result.get(key, "") for key in ranking_fields})

# ============================================================
# Print summary
# ============================================================

print("")
print("========== TOPOLOGY-AWARE RMSD ANALYSIS STATUS ==========")

status_counts = {}
for result in results:
    key = result.get("analysis_status", "")
    status_counts[key] = status_counts.get(key, 0) + 1

for key, value in status_counts.items():
    print(f"{key}: {value}")

print("")
print("========== FINAL REDOCKING VALIDATION TIERS ==========")

tier_counts = {}
for result in successful:
    key = result["redocking_validation_tier"]
    tier_counts[key] = tier_counts.get(key, 0) + 1

for key, value in tier_counts.items():
    print(f"{key}: {value}")

print("")
print("========== MD SELECTION BUCKETS ==========")

bucket_counts = {}
for result in successful:
    key = result["md_selection_bucket"]
    bucket_counts[key] = bucket_counts.get(key, 0) + 1

for key, value in bucket_counts.items():
    print(f"{key}: {value}")

print("")
print("========== TOP 15 FINAL PRIORITY ==========")

for rank, row in enumerate(ranked[:15], start=1):
    print(
        rank,
        row["receptor_id"],
        "| bucket:", row["md_selection_bucket"],
        "| tier:", row["redocking_validation_tier"],
        "| affinity:", row["vina_affinity_kcal_mol"],
        "| topology RMSD:", row["topology_aware_in_place_rmsd_A"],
        "| centroid:", row["centroid_shift_A"],
        "| recovery:", row["contact_recovery_percent"],
        "| new:", row["new_contact_percent_of_docked"]
    )

failed = [
    row for row in results
    if row.get("analysis_status") != "success"
]

if failed:
    print("")
    print("========== FAILED TOPOLOGY-AWARE ANALYSES ==========")
    for row in failed:
        print(row["receptor_id"], "|", row.get("error", ""))

print("")
print("Metrics saved to:", OUTPUT_METRICS)
print("Ranking saved to:", OUTPUT_RANKING)
