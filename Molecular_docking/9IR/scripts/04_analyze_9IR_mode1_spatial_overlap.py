from pathlib import Path
import csv
import math
import numpy as np
from rdkit import Chem

# ============================================================
# Paths
# ============================================================

ROOT = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\6_AutoDock_Vina\9IR")
FORMAL = ROOT / "Docking_48_Formal"

DOCKING_SDF_DIR = FORMAL / "outputs_sdf"
QC_DIR = FORMAL / "QC"

RECEPTOR_DIR = ROOT / "Aligned_Receptor_PDB"
REFERENCE_LIGAND_DIR = ROOT / "box_parameters" / "reference_ligands"

DOCKING_TABLE = QC_DIR / "vina_formal_docking_status_and_affinity_9IR.csv"

OUTPUT_METRICS = QC_DIR / "mode1_spatial_overlap_metrics_9IR.csv"
OUTPUT_RANKING = QC_DIR / "mode1_spatial_overlap_priority_ranking_9IR.csv"

CONTACT_CUTOFF_A = 4.5

QC_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Basic functions
# ============================================================

def dist(a, b):
    return float(np.linalg.norm(a - b))

def centroid_shift(coords_a, coords_b):
    return dist(coords_a.mean(axis=0), coords_b.mean(axis=0))

def parse_pdb_heavy_atoms(path):
    atoms = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()

            if record not in {"ATOM", "HETATM"}:
                continue

            try:
                atom_name = line[12:16].strip()
                resn = line[17:20].strip()
                chain = line[21].strip()
                resi = line[22:26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                element = line[76:78].strip().upper()
                if not element:
                    element = atom_name[0].upper()

            except Exception:
                continue

            if element == "H":
                continue

            atoms.append({
                "atom_name": atom_name,
                "resn": resn,
                "chain": chain,
                "resi": resi,
                "element": element,
                "coord": np.array([x, y, z], dtype=float)
            })

    return atoms

def read_reference_coords(reference_pdb):
    atoms = parse_pdb_heavy_atoms(reference_pdb)
    ligand_atoms = [atom for atom in atoms if atom["element"] != "H"]

    if len(ligand_atoms) != 10:
        raise RuntimeError(
            f"Reference 9IR should contain 10 heavy atoms, found {len(ligand_atoms)}"
        )

    if any(atom["element"] != "C" for atom in ligand_atoms):
        raise RuntimeError("Reference 9IR contains non-carbon heavy atoms.")

    return np.array([atom["coord"] for atom in ligand_atoms], dtype=float)

def read_docked_mode1_coords(sdf_file):
    supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
    mols = [mol for mol in supplier if mol is not None]

    if not mols:
        raise RuntimeError(f"No readable poses in {sdf_file}")

    mode1 = mols[0]
    conformer = mode1.GetConformer()

    heavy_atoms = [
        atom for atom in mode1.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]

    if len(heavy_atoms) != 10:
        raise RuntimeError(
            f"Docked 9IR should contain 10 heavy atoms, found {len(heavy_atoms)}"
        )

    if any(atom.GetSymbol() != "C" for atom in heavy_atoms):
        raise RuntimeError("Docked 9IR contains non-carbon heavy atoms.")

    coords = np.array([
        [
            conformer.GetAtomPosition(atom.GetIdx()).x,
            conformer.GetAtomPosition(atom.GetIdx()).y,
            conformer.GetAtomPosition(atom.GetIdx()).z
        ]
        for atom in heavy_atoms
    ], dtype=float)

    return coords, len(mols)

def contact_residue_set(receptor_atoms, ligand_coords, cutoff):
    contacts = set()

    for atom in receptor_atoms:
        if atom["element"] == "H":
            continue

        for ligand_coord in ligand_coords:
            if dist(atom["coord"], ligand_coord) <= cutoff:
                contacts.add((atom["chain"], atom["resi"], atom["resn"]))
                break

    return contacts

def format_residue_set(items):
    ordered = sorted(
        items,
        key=lambda x: (
            x[0],
            int(x[1]) if str(x[1]).isdigit() else str(x[1]),
            x[2]
        )
    )

    return ";".join(
        f"{chain}:{resi}:{resn}"
        for chain, resi, resn in ordered
    )

# ============================================================
# Fixed-frame optimal carbon assignment RMSD
# ============================================================

def spatial_assignment_rmsd(reference_coords, docked_coords):
    """
    Calculate a minimum one-to-one carbon assignment RMSD in the fixed
    receptor coordinate system.

    No rotation, translation or fitting is applied.
    This avoids dependence on PDB/SDF carbon atom naming order.

    Because 9IR is carbon-only, this is a spatial-overlap metric rather than
    a strict topology-mapped ligand RMSD.
    """

    n = len(reference_coords)

    if n != len(docked_coords):
        raise RuntimeError("Reference and docked coordinate counts differ.")

    cost = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            delta = docked_coords[i] - reference_coords[j]
            cost[i, j] = float(np.dot(delta, delta))

    # Dynamic programming over all one-to-one assignments.
    # For 10 atoms this is exact and fast: O(n * 2^n).
    dp = {0: (0.0, [])}

    for docked_index in range(n):
        new_dp = {}

        for used_mask, (current_cost, current_mapping) in dp.items():

            for reference_index in range(n):

                bit = 1 << reference_index

                if used_mask & bit:
                    continue

                new_mask = used_mask | bit
                new_cost = current_cost + cost[docked_index, reference_index]
                new_mapping = current_mapping + [(docked_index, reference_index)]

                if (
                    new_mask not in new_dp
                    or new_cost < new_dp[new_mask][0]
                ):
                    new_dp[new_mask] = (new_cost, new_mapping)

        dp = new_dp

    full_mask = (1 << n) - 1
    best_cost, best_mapping = dp[full_mask]

    rmsd = math.sqrt(best_cost / n)

    mapping_text = ";".join(
        f"dockedC{docked_i}->refC{ref_i}"
        for docked_i, ref_i in best_mapping
    )

    return rmsd, mapping_text

# ============================================================
# Read docking table
# ============================================================

with open(DOCKING_TABLE, "r", encoding="utf-8-sig", newline="") as handle:
    docking_rows = list(csv.DictReader(handle))

completed_rows = [
    row for row in docking_rows
    if row.get("docking_status") == "completed"
]

print("Completed docking records found:", len(completed_rows))

if len(completed_rows) != 48:
    raise RuntimeError(
        f"Expected 48 completed docking records, found {len(completed_rows)}"
    )

results = []

# ============================================================
# Analyze 48 candidates
# ============================================================

for row in completed_rows:

    receptor_id = row["receptor_id"]

    receptor_pdb = RECEPTOR_DIR / f"{receptor_id}_aligned.pdb"
    reference_pdb = REFERENCE_LIGAND_DIR / f"{receptor_id}_9IR_reference.pdb"
    docked_sdf = DOCKING_SDF_DIR / f"{receptor_id}_9IR_out.sdf"

    try:
        receptor_atoms = parse_pdb_heavy_atoms(receptor_pdb)
        reference_coords = read_reference_coords(reference_pdb)
        docked_coords, pose_count = read_docked_mode1_coords(docked_sdf)

        assignment_rmsd, assignment_mapping = spatial_assignment_rmsd(
            reference_coords,
            docked_coords
        )

        centroid_distance = centroid_shift(reference_coords, docked_coords)

        reference_contacts = contact_residue_set(
            receptor_atoms,
            reference_coords,
            CONTACT_CUTOFF_A
        )

        docked_contacts = contact_residue_set(
            receptor_atoms,
            docked_coords,
            CONTACT_CUTOFF_A
        )

        recovered_contacts = reference_contacts & docked_contacts
        lost_contacts = reference_contacts - docked_contacts
        new_contacts = docked_contacts - reference_contacts
        union_contacts = reference_contacts | docked_contacts

        recovery_percent = (
            100.0 * len(recovered_contacts) / len(reference_contacts)
            if reference_contacts else 0.0
        )

        new_contact_percent = (
            100.0 * len(new_contacts) / len(docked_contacts)
            if docked_contacts else 0.0
        )

        jaccard = (
            len(recovered_contacts) / len(union_contacts)
            if union_contacts else 0.0
        )

        affinity = float(row["best_affinity_kcal_mol"])
        pocket_rmsd = float(row["alignment_rmsd_A"])
        pocket_risk = row["pocket_alignment_risk"]

        # ====================================================
        # Screening tiers
        # ====================================================

        if (
            assignment_rmsd <= 1.5
            and centroid_distance <= 1.0
            and recovery_percent >= 80.0
            and new_contact_percent <= 20.0
            and pocket_risk == "Acceptable_pocket_alignment"
        ):
            tier = "Tier_A_high_confidence_recovered_pose"

        elif (
            assignment_rmsd <= 2.5
            and centroid_distance <= 2.0
            and recovery_percent >= 60.0
            and new_contact_percent <= 35.0
        ):
            tier = "Tier_B_supported_pose"

        else:
            tier = "Tier_C_low_priority_pose"

        results.append({
            "receptor_id": receptor_id,
            "analysis_status": "success",
            "vina_affinity_kcal_mol": round(affinity, 4),
            "pocket_alignment_rmsd_A": round(pocket_rmsd, 4),
            "pocket_alignment_risk": pocket_risk,
            "poses_in_sdf": pose_count,
            "spatial_assignment_rmsd_A": round(assignment_rmsd, 4),
            "centroid_shift_A": round(centroid_distance, 4),
            "contact_cutoff_A": CONTACT_CUTOFF_A,
            "reference_contact_residue_count": len(reference_contacts),
            "docked_contact_residue_count": len(docked_contacts),
            "recovered_contact_residue_count": len(recovered_contacts),
            "lost_contact_residue_count": len(lost_contacts),
            "new_contact_residue_count": len(new_contacts),
            "contact_recovery_percent": round(recovery_percent, 4),
            "new_contact_percent_of_docked": round(new_contact_percent, 4),
            "contact_jaccard": round(jaccard, 4),
            "reference_contact_residues": format_residue_set(reference_contacts),
            "docked_contact_residues": format_residue_set(docked_contacts),
            "recovered_contact_residues": format_residue_set(recovered_contacts),
            "new_contact_residues": format_residue_set(new_contacts),
            "spatial_assignment_mapping": assignment_mapping,
            "redocking_validation_tier": tier
        })

    except Exception as exc:
        results.append({
            "receptor_id": receptor_id,
            "analysis_status": "analysis_failed",
            "error": str(exc)
        })

# ============================================================
# Output full metrics
# ============================================================

fieldnames = [
    "receptor_id",
    "analysis_status",
    "vina_affinity_kcal_mol",
    "pocket_alignment_rmsd_A",
    "pocket_alignment_risk",
    "poses_in_sdf",
    "spatial_assignment_rmsd_A",
    "centroid_shift_A",
    "contact_cutoff_A",
    "reference_contact_residue_count",
    "docked_contact_residue_count",
    "recovered_contact_residue_count",
    "lost_contact_residue_count",
    "new_contact_residue_count",
    "contact_recovery_percent",
    "new_contact_percent_of_docked",
    "contact_jaccard",
    "reference_contact_residues",
    "docked_contact_residues",
    "recovered_contact_residues",
    "new_contact_residues",
    "spatial_assignment_mapping",
    "redocking_validation_tier",
    "error"
]

with open(OUTPUT_METRICS, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    for result in results:
        writer.writerow({key: result.get(key, "") for key in fieldnames})

# ============================================================
# Ranking
# ============================================================

successful = [
    row for row in results
    if row.get("analysis_status") == "success"
]

tier_order = {
    "Tier_A_high_confidence_recovered_pose": 0,
    "Tier_B_supported_pose": 1,
    "Tier_C_low_priority_pose": 2
}

successful_sorted = sorted(
    successful,
    key=lambda row: (
        tier_order.get(row["redocking_validation_tier"], 9),
        -float(row["contact_recovery_percent"]),
        float(row["new_contact_percent_of_docked"]),
        float(row["spatial_assignment_rmsd_A"]),
        float(row["centroid_shift_A"]),
        float(row["vina_affinity_kcal_mol"])
    )
)

ranking_fields = [
    "receptor_id",
    "redocking_validation_tier",
    "vina_affinity_kcal_mol",
    "spatial_assignment_rmsd_A",
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

    for result in successful_sorted:
        writer.writerow({key: result.get(key, "") for key in ranking_fields})

# ============================================================
# Print summary
# ============================================================

print("")
print("========== SPATIAL OVERLAP ANALYSIS STATUS ==========")

status_counts = {}
for result in results:
    status = result.get("analysis_status", "")
    status_counts[status] = status_counts.get(status, 0) + 1

for status, count in status_counts.items():
    print(f"{status}: {count}")

print("")
print("========== SPATIAL OVERLAP VALIDATION TIERS ==========")

tier_counts = {}
for result in successful:
    tier = result["redocking_validation_tier"]
    tier_counts[tier] = tier_counts.get(tier, 0) + 1

for tier, count in tier_counts.items():
    print(f"{tier}: {count}")

print("")
print("========== TOP 15 BY REDOCKING PRIORITY ==========")

for rank, row in enumerate(successful_sorted[:15], start=1):
    print(
        rank,
        row["receptor_id"],
        "| tier:", row["redocking_validation_tier"],
        "| affinity:", row["vina_affinity_kcal_mol"],
        "| spatial RMSD:", row["spatial_assignment_rmsd_A"],
        "| centroid shift:", row["centroid_shift_A"],
        "| recovery:", row["contact_recovery_percent"],
        "| new contacts:", row["new_contact_percent_of_docked"]
    )

print("")
print("Full metrics saved to:", OUTPUT_METRICS)
print("Ranking saved to:", OUTPUT_RANKING)
