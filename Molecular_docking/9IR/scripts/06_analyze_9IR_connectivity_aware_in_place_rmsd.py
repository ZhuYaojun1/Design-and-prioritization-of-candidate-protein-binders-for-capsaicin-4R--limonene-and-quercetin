from pathlib import Path
import csv
from rdkit import Chem, RDLogger
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

OUTPUT_METRICS = QC_DIR / "mode1_connectivity_aware_in_place_rmsd_metrics_9IR.csv"
OUTPUT_RANKING = QC_DIR / "mode1_connectivity_aware_MD_priority_ranking_9IR.csv"

# AssignBondOrdersFromTemplate may report multiple equivalent mappings
# for this carbon-only ligand. These warnings are expected here because
# final RMSD is calculated using CalcRMS with legal topology matching.
RDLogger.DisableLog("rdApp.warning")

# ============================================================
# Molecule utilities
# ============================================================

def read_first_sdf_molecule(path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    raise RuntimeError(f"Unable to read molecule from: {path}")

def make_achiral_heavy_molecule(mol):
    """
    Remove hydrogens and unreliable stereo labels for the reference-pose
    comparison. Connectivity and coordinates are retained.
    """
    heavy = Chem.RemoveHs(mol)
    Chem.RemoveStereochemistry(heavy)
    Chem.SanitizeMol(heavy)
    return heavy

template = make_achiral_heavy_molecule(
    read_first_sdf_molecule(TEMPLATE_SDF)
)

template_connectivity = Chem.MolToSmiles(
    template,
    isomericSmiles=False
)

print("Template connectivity used for reference matching:")
print(template_connectivity)

def build_reference_molecule(reference_pdb):
    """
    Construct a reference ligand molecule using coordinates from the
    RFD3-derived PDB while assigning bond orders from the 9IR template.

    Stereo labels are not used because the PDB-derived reference pose does
    not reliably preserve the original chiral annotation.
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
    Chem.RemoveStereochemistry(raw)

    try:
        reference = AllChem.AssignBondOrdersFromTemplate(template, raw)
        Chem.RemoveStereochemistry(reference)
        Chem.SanitizeMol(reference)
    except Exception as exc:
        raise RuntimeError(
            f"Reference connectivity assignment failed for "
            f"{reference_pdb.name}: {exc}"
        )

    reference_connectivity = Chem.MolToSmiles(
        reference,
        isomericSmiles=False
    )

    if reference_connectivity != template_connectivity:
        raise RuntimeError(
            f"Reference connectivity mismatch: "
            f"{reference_connectivity} != {template_connectivity}"
        )

    return reference

def read_docked_mode1_molecule(docked_sdf):
    """
    The first molecule in each Meeko-exported SDF corresponds to Vina mode 1.
    """
    docked = make_achiral_heavy_molecule(
        read_first_sdf_molecule(docked_sdf)
    )

    docked_connectivity = Chem.MolToSmiles(
        docked,
        isomericSmiles=False
    )

    if docked_connectivity != template_connectivity:
        raise RuntimeError(
            f"Docked connectivity mismatch: "
            f"{docked_connectivity} != {template_connectivity}"
        )

    return docked

# ============================================================
# Read previous contact and docking metrics
# ============================================================

with open(PREVIOUS_METRICS, "r", encoding="utf-8-sig", newline="") as handle:
    previous_rows = list(csv.DictReader(handle))

previous_success = [
    row for row in previous_rows
    if row.get("analysis_status") == "success"
]

print("")
print("Previous successful contact-metric records found:", len(previous_success))

if len(previous_success) != 48:
    raise RuntimeError(
        f"Expected 48 previous successful rows, found {len(previous_success)}"
    )

results = []

# ============================================================
# Connectivity-aware in-place RMSD analysis
# ============================================================

for row in previous_success:

    receptor_id = row["receptor_id"]

    reference_pdb = REFERENCE_DIR / f"{receptor_id}_9IR_reference.pdb"
    docked_sdf = DOCKING_SDF_DIR / f"{receptor_id}_9IR_out.sdf"

    try:
        reference = build_reference_molecule(reference_pdb)
        docked = read_docked_mode1_molecule(docked_sdf)

        # CalcRMS calculates RMSD in the existing coordinate frame:
        # no ligand superposition is applied before calculating RMSD.
        # Legal topology/symmetry-related mappings are considered.
        in_place_rmsd = rdMolAlign.CalcRMS(
            docked,
            reference,
            maxMatches=100000
        )

        affinity = float(row["vina_affinity_kcal_mol"])
        centroid_shift = float(row["centroid_shift_A"])
        contact_recovery = float(row["contact_recovery_percent"])
        new_contact_percent = float(row["new_contact_percent_of_docked"])
        contact_jaccard = float(row["contact_jaccard"])
        pocket_alignment_rmsd = float(row["pocket_alignment_rmsd_A"])
        pocket_risk = row["pocket_alignment_risk"]

        # ====================================================
        # Final redocking validation tiers
        # ====================================================

        if (
            in_place_rmsd <= 2.0
            and centroid_shift <= 1.0
            and contact_recovery >= 80.0
            and new_contact_percent <= 20.0
            and pocket_risk == "Acceptable_pocket_alignment"
        ):
            tier = "Tier_A_connectivity_supported_pose"

        elif (
            in_place_rmsd <= 3.0
            and centroid_shift <= 2.0
            and contact_recovery >= 60.0
            and new_contact_percent <= 35.0
        ):
            tier = "Tier_B_partially_supported_pose"

        else:
            tier = "Tier_C_low_priority_pose"

        # ====================================================
        # Initial MD prioritization
        # ====================================================

        if (
            tier == "Tier_A_connectivity_supported_pose"
            and affinity <= -6.8
        ):
            md_bucket = "Primary_MD_shortlist"

        elif tier == "Tier_A_connectivity_supported_pose":
            md_bucket = "Pose_supported_backup"

        elif (
            tier == "Tier_B_partially_supported_pose"
            and affinity <= -7.0
        ):
            md_bucket = "Strong_affinity_manual_review"

        else:
            md_bucket = "Not_priority_for_initial_MD"

        results.append({
            "receptor_id": receptor_id,
            "analysis_status": "success",
            "vina_affinity_kcal_mol": round(affinity, 4),
            "connectivity_aware_in_place_rmsd_A": round(float(in_place_rmsd), 4),
            "centroid_shift_A": round(centroid_shift, 4),
            "contact_recovery_percent": round(contact_recovery, 4),
            "new_contact_percent_of_docked": round(new_contact_percent, 4),
            "contact_jaccard": round(contact_jaccard, 4),
            "pocket_alignment_rmsd_A": round(pocket_alignment_rmsd, 4),
            "pocket_alignment_risk": pocket_risk,
            "redocking_validation_tier": tier,
            "md_selection_bucket": md_bucket,
            "reference_contact_residues": row.get("reference_contact_residues", ""),
            "docked_contact_residues": row.get("docked_contact_residues", ""),
            "recovered_contact_residues": row.get("recovered_contact_residues", ""),
            "new_contact_residues": row.get("new_contact_residues", ""),
            "stereo_handling_note": (
                "RMSD calculated in fixed receptor coordinates using "
                "connectivity-aware matching after removing unreliable "
                "stereo labels from PDB-derived reference ligand."
            )
        })

    except Exception as exc:
        results.append({
            "receptor_id": receptor_id,
            "analysis_status": "analysis_failed",
            "error": str(exc)
        })

# ============================================================
# Write full metrics table
# ============================================================

metric_fields = [
    "receptor_id",
    "analysis_status",
    "vina_affinity_kcal_mol",
    "connectivity_aware_in_place_rmsd_A",
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
    "stereo_handling_note",
    "error"
]

with open(OUTPUT_METRICS, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=metric_fields)
    writer.writeheader()

    for result in results:
        writer.writerow({key: result.get(key, "") for key in metric_fields})

# ============================================================
# Rank successful candidates
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
    "Tier_A_connectivity_supported_pose": 0,
    "Tier_B_partially_supported_pose": 1,
    "Tier_C_low_priority_pose": 2
}

ranked = sorted(
    successful,
    key=lambda row: (
        bucket_order.get(row["md_selection_bucket"], 9),
        tier_order.get(row["redocking_validation_tier"], 9),
        float(row["vina_affinity_kcal_mol"]),
        float(row["connectivity_aware_in_place_rmsd_A"]),
        -float(row["contact_recovery_percent"]),
        float(row["new_contact_percent_of_docked"])
    )
)

ranking_fields = [
    "receptor_id",
    "md_selection_bucket",
    "redocking_validation_tier",
    "vina_affinity_kcal_mol",
    "connectivity_aware_in_place_rmsd_A",
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
print("========== CONNECTIVITY-AWARE RMSD ANALYSIS STATUS ==========")

status_counts = {}
for result in results:
    status = result.get("analysis_status", "")
    status_counts[status] = status_counts.get(status, 0) + 1

for status, count in status_counts.items():
    print(f"{status}: {count}")

print("")
print("========== FINAL REDOCKING VALIDATION TIERS ==========")

tier_counts = {}
for result in successful:
    tier = result["redocking_validation_tier"]
    tier_counts[tier] = tier_counts.get(tier, 0) + 1

for tier, count in tier_counts.items():
    print(f"{tier}: {count}")

print("")
print("========== MD SELECTION BUCKETS ==========")

bucket_counts = {}
for result in successful:
    bucket = result["md_selection_bucket"]
    bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

for bucket, count in bucket_counts.items():
    print(f"{bucket}: {count}")

print("")
print("========== TOP 15 FINAL PRIORITY ==========")

for rank, row in enumerate(ranked[:15], start=1):
    print(
        rank,
        row["receptor_id"],
        "| bucket:", row["md_selection_bucket"],
        "| tier:", row["redocking_validation_tier"],
        "| affinity:", row["vina_affinity_kcal_mol"],
        "| in-place RMSD:", row["connectivity_aware_in_place_rmsd_A"],
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
    print("========== FAILED ANALYSES ==========")
    for row in failed:
        print(row["receptor_id"], "|", row.get("error", ""))

print("")
print("Metrics saved to:", OUTPUT_METRICS)
print("Ranking saved to:", OUTPUT_RANKING)
