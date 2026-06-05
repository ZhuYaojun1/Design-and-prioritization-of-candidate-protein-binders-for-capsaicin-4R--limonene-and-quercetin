from pathlib import Path
import csv
import math
import re
import numpy as np
from rdkit import Chem

# ============================================================
# Paths and parameters
# ============================================================

ROOT = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\6_AutoDock_Vina\9IR")
FORMAL = ROOT / "Docking_48_Formal"

DOCKING_SDF_DIR = FORMAL / "outputs_sdf"
QC_DIR = FORMAL / "QC"

RECEPTOR_DIR = ROOT / "Aligned_Receptor_PDB"
REFERENCE_LIGAND_DIR = ROOT / "box_parameters" / "reference_ligands"

DOCKING_TABLE = QC_DIR / "vina_formal_docking_status_and_affinity_9IR.csv"
IDEAL_SDF = ROOT / "9IR_PDBQT" / "9IR_ideal.sdf"

OUTPUT_METRICS = QC_DIR / "mode1_redocking_pose_contact_metrics_9IR.csv"
OUTPUT_RANKING = QC_DIR / "mode1_redocking_priority_ranking_9IR.csv"

CONTACT_CUTOFF_A = 4.5

# RFD3 reference 9IR atom names confirmed in PyMOL:
# C, C1, C2, ... C9
REFERENCE_ATOM_ORDER = ["C"] + [f"C{i}" for i in range(1, 10)]

QC_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Utility functions
# ============================================================

def euclidean(a, b):
    return float(np.linalg.norm(a - b))

def direct_rmsd(coords_a, coords_b):
    if coords_a.shape != coords_b.shape:
        raise ValueError(f"Coordinate shapes differ: {coords_a.shape} vs {coords_b.shape}")
    return float(np.sqrt(np.mean(np.sum((coords_a - coords_b) ** 2, axis=1))))

def centroid_shift(coords_a, coords_b):
    return euclidean(coords_a.mean(axis=0), coords_b.mean(axis=0))

def kabsch_aligned_rmsd(coords_a, coords_b):
    """
    Used only to check whether the assumed atom order of reference PDB
    corresponds to the atom order of 9IR_ideal.sdf.
    This is NOT used for docking pose recovery ranking.
    """
    pa = coords_a - coords_a.mean(axis=0)
    pb = coords_b - coords_b.mean(axis=0)

    covariance = pa.T @ pb
    u, s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T

    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T

    aligned = pa @ rotation
    return float(np.sqrt(np.mean(np.sum((aligned - pb) ** 2, axis=1))))

def parse_pdb_atoms(path, keep_records=("ATOM", "HETATM")):
    atoms = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()

            if record not in keep_records:
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
                    element = re.sub(r"[^A-Za-z]", "", atom_name)[0].upper()
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

def read_reference_ligand_coords(reference_pdb):
    atoms = parse_pdb_atoms(reference_pdb, keep_records=("ATOM", "HETATM"))

    atom_by_name = {}
    for atom in atoms:
        atom_by_name[atom["atom_name"]] = atom

    missing = [name for name in REFERENCE_ATOM_ORDER if name not in atom_by_name]
    if missing:
        raise RuntimeError(
            f"Reference ligand missing expected atoms {missing}: {reference_pdb}"
        )

    coords = np.array(
        [atom_by_name[name]["coord"] for name in REFERENCE_ATOM_ORDER],
        dtype=float
    )
    return coords

def read_sdf_mode1_heavy_coords(sdf_file):
    supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False)

    mols = [mol for mol in supplier if mol is not None]

    if not mols:
        raise RuntimeError(f"No readable molecule found in {sdf_file}")

    mode1 = mols[0]
    conformer = mode1.GetConformer()

    heavy_indices = [
        atom.GetIdx()
        for atom in mode1.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]

    coords = np.array(
        [
            [
                conformer.GetAtomPosition(idx).x,
                conformer.GetAtomPosition(idx).y,
                conformer.GetAtomPosition(idx).z
            ]
            for idx in heavy_indices
        ],
        dtype=float
    )

    return coords, len(mols), len(heavy_indices)

def read_ideal_heavy_coords():
    supplier = Chem.SDMolSupplier(str(IDEAL_SDF), removeHs=False)
    mol = next((m for m in supplier if m is not None), None)

    if mol is None:
        raise RuntimeError(f"Unable to read ideal ligand: {IDEAL_SDF}")

    conformer = mol.GetConformer()

    heavy_indices = [
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() > 1
    ]

    coords = np.array(
        [
            [
                conformer.GetAtomPosition(idx).x,
                conformer.GetAtomPosition(idx).y,
                conformer.GetAtomPosition(idx).z
            ]
            for idx in heavy_indices
        ],
        dtype=float
    )

    return coords

def contact_residue_set(receptor_atoms, ligand_coords, cutoff):
    contacts = set()

    for atom in receptor_atoms:
        for ligand_coord in ligand_coords:
            if euclidean(atom["coord"], ligand_coord) <= cutoff:
                contacts.add((atom["chain"], atom["resi"], atom["resn"]))
                break

    return contacts

def format_residue_set(residue_set):
    ordered = sorted(
        residue_set,
        key=lambda x: (x[0], int(x[1]) if str(x[1]).isdigit() else str(x[1]), x[2])
    )
    return ";".join(f"{chain}:{resi}:{resn}" for chain, resi, resn in ordered)

# ============================================================
# Read global inputs
# ============================================================

ideal_coords = read_ideal_heavy_coords()

with open(DOCKING_TABLE, "r", encoding="utf-8-sig", newline="") as handle:
    docking_rows = list(csv.DictReader(handle))

completed_rows = [
    row for row in docking_rows
    if row.get("docking_status") == "completed"
]

print(f"Completed docking records found: {len(completed_rows)}")

if len(completed_rows) != 48:
    raise RuntimeError(
        f"Expected 48 completed docking records, found {len(completed_rows)}"
    )

results = []

# ============================================================
# Analyze each candidate
# ============================================================

for row in completed_rows:

    receptor_id = row["receptor_id"]

    receptor_pdb = RECEPTOR_DIR / f"{receptor_id}_aligned.pdb"
    reference_pdb = REFERENCE_LIGAND_DIR / f"{receptor_id}_9IR_reference.pdb"
    docked_sdf = DOCKING_SDF_DIR / f"{receptor_id}_9IR_out.sdf"

    if not receptor_pdb.exists() or not reference_pdb.exists() or not docked_sdf.exists():
        results.append({
            "receptor_id": receptor_id,
            "analysis_status": "missing_required_file"
        })
        continue

    try:
        receptor_atoms = parse_pdb_atoms(receptor_pdb, keep_records=("ATOM",))
        reference_coords = read_reference_ligand_coords(reference_pdb)
        docked_coords, pose_count, docked_heavy_atom_count = read_sdf_mode1_heavy_coords(docked_sdf)

        if reference_coords.shape[0] != 10 or docked_heavy_atom_count != 10:
            raise RuntimeError(
                f"Unexpected heavy atom count: reference={reference_coords.shape[0]}, "
                f"docked={docked_heavy_atom_count}"
            )

        # This validates the assumed RFD3 atom-name order against the 9IR ideal SDF atom order.
        # A small value supports use of ordered direct RMSD below.
        reference_mapping_fit_rmsd = kabsch_aligned_rmsd(ideal_coords, reference_coords)

        if reference_mapping_fit_rmsd <= 0.25:
            mapping_status = "Validated_by_ideal_geometry"
            pose_rmsd = direct_rmsd(docked_coords, reference_coords)
        else:
            mapping_status = "Check_atom_mapping_before_using_pose_RMSD"
            pose_rmsd = ""

        centroid_distance = centroid_shift(docked_coords, reference_coords)

        reference_contacts = contact_residue_set(
            receptor_atoms, reference_coords, CONTACT_CUTOFF_A
        )
        docked_contacts = contact_residue_set(
            receptor_atoms, docked_coords, CONTACT_CUTOFF_A
        )

        recovered_contacts = reference_contacts & docked_contacts
        lost_contacts = reference_contacts - docked_contacts
        new_contacts = docked_contacts - reference_contacts
        union_contacts = reference_contacts | docked_contacts

        recovery_percent = (
            100.0 * len(recovered_contacts) / len(reference_contacts)
            if reference_contacts else ""
        )

        new_contact_percent = (
            100.0 * len(new_contacts) / len(docked_contacts)
            if docked_contacts else ""
        )

        jaccard = (
            len(recovered_contacts) / len(union_contacts)
            if union_contacts else ""
        )

        affinity = float(row["best_affinity_kcal_mol"])
        alignment_rmsd = float(row["alignment_rmsd_A"])

        if mapping_status != "Validated_by_ideal_geometry":
            tier = "Mapping_check_required"
        elif pose_rmsd <= 2.0 and recovery_percent >= 70.0 and new_contact_percent <= 30.0:
            tier = "Tier_A_pose_recovered"
        elif pose_rmsd <= 3.0 and recovery_percent >= 50.0:
            tier = "Tier_B_partial_recovery"
        else:
            tier = "Tier_C_poor_pose_recovery"

        results.append({
            "receptor_id": receptor_id,
            "analysis_status": "success",
            "vina_affinity_kcal_mol": round(affinity, 4),
            "pocket_alignment_rmsd_A": round(alignment_rmsd, 4),
            "pocket_alignment_risk": row["pocket_alignment_risk"],
            "poses_in_sdf": pose_count,
            "reference_mapping_fit_rmsd_A": round(reference_mapping_fit_rmsd, 4),
            "mapping_status": mapping_status,
            "mode1_pose_rmsd_A": round(pose_rmsd, 4) if pose_rmsd != "" else "",
            "centroid_shift_A": round(centroid_distance, 4),
            "contact_cutoff_A": CONTACT_CUTOFF_A,
            "reference_contact_residue_count": len(reference_contacts),
            "docked_contact_residue_count": len(docked_contacts),
            "recovered_contact_residue_count": len(recovered_contacts),
            "lost_contact_residue_count": len(lost_contacts),
            "new_contact_residue_count": len(new_contacts),
            "contact_recovery_percent": round(recovery_percent, 4) if recovery_percent != "" else "",
            "new_contact_percent_of_docked": round(new_contact_percent, 4) if new_contact_percent != "" else "",
            "contact_jaccard": round(jaccard, 4) if jaccard != "" else "",
            "reference_contact_residues": format_residue_set(reference_contacts),
            "docked_contact_residues": format_residue_set(docked_contacts),
            "recovered_contact_residues": format_residue_set(recovered_contacts),
            "new_contact_residues": format_residue_set(new_contacts),
            "redocking_validation_tier": tier
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

fieldnames = [
    "receptor_id",
    "analysis_status",
    "vina_affinity_kcal_mol",
    "pocket_alignment_rmsd_A",
    "pocket_alignment_risk",
    "poses_in_sdf",
    "reference_mapping_fit_rmsd_A",
    "mapping_status",
    "mode1_pose_rmsd_A",
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
    "redocking_validation_tier",
    "error"
]

with open(OUTPUT_METRICS, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    for result in results:
        writer.writerow({key: result.get(key, "") for key in fieldnames})

# ============================================================
# Write ranking table
# ============================================================

successful = [
    row for row in results
    if row.get("analysis_status") == "success"
]

tier_order = {
    "Tier_A_pose_recovered": 0,
    "Tier_B_partial_recovery": 1,
    "Tier_C_poor_pose_recovery": 2,
    "Mapping_check_required": 3
}

successful_sorted = sorted(
    successful,
    key=lambda row: (
        tier_order.get(row["redocking_validation_tier"], 9),
        float(row["mode1_pose_rmsd_A"]) if row["mode1_pose_rmsd_A"] != "" else 999.0,
        -float(row["contact_recovery_percent"]) if row["contact_recovery_percent"] != "" else 999.0,
        float(row["vina_affinity_kcal_mol"])
    )
)

ranking_fields = [
    "receptor_id",
    "redocking_validation_tier",
    "vina_affinity_kcal_mol",
    "mode1_pose_rmsd_A",
    "centroid_shift_A",
    "contact_recovery_percent",
    "new_contact_percent_of_docked",
    "contact_jaccard",
    "pocket_alignment_rmsd_A",
    "pocket_alignment_risk",
    "mapping_status"
]

with open(OUTPUT_RANKING, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=ranking_fields)
    writer.writeheader()

    for result in successful_sorted:
        writer.writerow({key: result.get(key, "") for key in ranking_fields})

# ============================================================
# Console summary
# ============================================================

print("")
print("========== MODE 1 REDOCKING ANALYSIS STATUS ==========")

status_counts = {}
for result in results:
    status = result.get("analysis_status", "")
    status_counts[status] = status_counts.get(status, 0) + 1

for key, value in status_counts.items():
    print(f"{key}: {value}")

tier_counts = {}
for result in successful:
    tier = result.get("redocking_validation_tier", "")
    tier_counts[tier] = tier_counts.get(tier, 0) + 1

print("")
print("========== REDOCKING VALIDATION TIERS ==========")

for key, value in tier_counts.items():
    print(f"{key}: {value}")

mapping_fail = [
    row for row in successful
    if row["mapping_status"] != "Validated_by_ideal_geometry"
]

print("")
print(f"Candidates requiring atom mapping check: {len(mapping_fail)}")

print("")
print("========== TOP 10 BY POSE-RECOVERY PRIORITY ==========")

for rank, row in enumerate(successful_sorted[:10], start=1):
    print(
        rank,
        row["receptor_id"],
        "| tier:", row["redocking_validation_tier"],
        "| affinity:", row["vina_affinity_kcal_mol"],
        "| pose RMSD:", row["mode1_pose_rmsd_A"],
        "| recovery:", row["contact_recovery_percent"],
        "| new contacts:", row["new_contact_percent_of_docked"]
    )

print("")
print(f"Full metrics saved to: {OUTPUT_METRICS}")
print(f"Ranking saved to: {OUTPUT_RANKING}")
