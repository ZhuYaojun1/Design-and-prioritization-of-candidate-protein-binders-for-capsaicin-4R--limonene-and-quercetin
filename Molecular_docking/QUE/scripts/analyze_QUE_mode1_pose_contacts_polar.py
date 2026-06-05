from pathlib import Path
import csv
import math
import numpy as np

from rdkit import Chem, RDLogger, RDConfig
from rdkit.Chem import AllChem, ChemicalFeatures, rdMolAlign

# ============================================================
# Paths and parameters
# ============================================================

ROOT = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\6_AutoDock_Vina\QUE")
FORMAL = ROOT / "Docking_30_Formal"

QC_DIR = FORMAL / "QC"
DOCKING_SDF_DIR = FORMAL / "outputs_sdf"

RECEPTOR_DIR = ROOT / "Aligned_Receptor_PDB"
REFERENCE_LIGAND_DIR = ROOT / "box_parameters" / "reference_ligands"

DOCKING_TABLE = QC_DIR / "vina_formal_docking_status_and_affinity_QUE.csv"
TEMPLATE_SDF = ROOT / "QUE_PDBQT" / "QUE_ideal.sdf"

OUTPUT_METRICS = QC_DIR / "mode1_connectivity_polar_contact_metrics_QUE.csv"
OUTPUT_RANKING = QC_DIR / "mode1_connectivity_polar_contact_priority_ranking_QUE.csv"

CONTACT_CUTOFF_A = 4.5
POLAR_CONTACT_CUTOFF_A = 3.5

QC_DIR.mkdir(parents=True, exist_ok=True)

RDLogger.DisableLog("rdApp.warning")

# ============================================================
# General utilities
# ============================================================

def distance(coord_a, coord_b):
    return float(np.linalg.norm(coord_a - coord_b))

def percentage(numerator, denominator):
    if denominator == 0:
        return ""
    return 100.0 * numerator / denominator

def molecule_coordinates(mol):
    conformer = mol.GetConformer()
    return np.array(
        [
            [
                conformer.GetAtomPosition(atom.GetIdx()).x,
                conformer.GetAtomPosition(atom.GetIdx()).y,
                conformer.GetAtomPosition(atom.GetIdx()).z
            ]
            for atom in mol.GetAtoms()
        ],
        dtype=float
    )

def centroid_shift(coords_a, coords_b):
    return distance(coords_a.mean(axis=0), coords_b.mean(axis=0))

def format_items(items):
    return ";".join(sorted(items))

# ============================================================
# Read and prepare ligand molecules
# ============================================================

def read_first_sdf_molecule(path):
    supplier = Chem.SDMolSupplier(str(path), removeHs=False)
    for mol in supplier:
        if mol is not None:
            return mol
    raise RuntimeError(f"Unable to read molecule from: {path}")

def make_achiral_heavy_molecule(mol):
    heavy = Chem.RemoveHs(mol, sanitize=False)
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

template_heavy_atom_count = template.GetNumAtoms()

print("QUE template connectivity:", template_connectivity)
print("QUE template heavy atom count:", template_heavy_atom_count)

if template_heavy_atom_count != 22:
    raise RuntimeError(
        f"Expected QUE to contain 22 heavy atoms, found {template_heavy_atom_count}."
    )

feature_factory = ChemicalFeatures.BuildFeatureFactory(
    str(Path(RDConfig.RDDataDir) / "BaseFeatures.fdef")
)

def ligand_feature_atom_ids(mol, family):
    atom_ids = set()
    for feature in feature_factory.GetFeaturesForMol(mol):
        if feature.GetFamily() == family:
            atom_ids.update(feature.GetAtomIds())
    return atom_ids

def build_reference_ligand(reference_pdb):
    """
    Coordinates are retained from the RFdiffusion3-derived reference PDB.
    Bond orders are assigned from QUE_ideal.sdf so that legal topology-based
    in-place RMSD comparison can be performed.
    """
    raw = Chem.MolFromPDBFile(
        str(reference_pdb),
        sanitize=False,
        removeHs=False,
        proximityBonding=True
    )

    if raw is None:
        raise RuntimeError(f"Unable to read reference ligand PDB: {reference_pdb}")

    raw = Chem.RemoveHs(raw, sanitize=False)
    Chem.RemoveStereochemistry(raw)

    reference = AllChem.AssignBondOrdersFromTemplate(template, raw)
    Chem.RemoveStereochemistry(reference)
    Chem.SanitizeMol(reference)

    reference_connectivity = Chem.MolToSmiles(
        reference,
        isomericSmiles=False
    )

    if reference_connectivity != template_connectivity:
        raise RuntimeError(
            f"Reference connectivity mismatch for {reference_pdb.name}: "
            f"{reference_connectivity} != {template_connectivity}"
        )

    if reference.GetNumAtoms() != 22:
        raise RuntimeError(
            f"Reference QUE heavy atom count is {reference.GetNumAtoms()}, expected 22."
        )

    return reference

def read_docked_mode1(docking_sdf):
    """
    The first molecule in the Meeko-exported SDF is treated as Vina mode 1.
    """
    supplier = Chem.SDMolSupplier(str(docking_sdf), removeHs=False)
    poses = [mol for mol in supplier if mol is not None]

    if not poses:
        raise RuntimeError(f"No readable docking pose in: {docking_sdf}")

    mode1 = make_achiral_heavy_molecule(poses[0])

    docked_connectivity = Chem.MolToSmiles(
        mode1,
        isomericSmiles=False
    )

    if docked_connectivity != template_connectivity:
        raise RuntimeError(
            f"Docked connectivity mismatch for {docking_sdf.name}: "
            f"{docked_connectivity} != {template_connectivity}"
        )

    if mode1.GetNumAtoms() != 22:
        raise RuntimeError(
            f"Docked QUE heavy atom count is {mode1.GetNumAtoms()}, expected 22."
        )

    return mode1, len(poses)

# ============================================================
# Read receptor atoms
# ============================================================

def parse_receptor_atoms(pdb_file):
    atoms = []

    with open(pdb_file, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line[:6].strip() != "ATOM":
                continue

            try:
                atom_name = line[12:16].strip()
                resn = line[17:20].strip().upper()
                chain = line[21].strip()
                resi = line[22:26].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                element = line[76:78].strip().upper()
                if not element:
                    letters = "".join(c for c in atom_name if c.isalpha())
                    element = letters[0].upper()

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

def residue_key(atom):
    return f"{atom['chain']}:{atom['resi']}:{atom['resn']}"

# ============================================================
# General heavy-atom contact analysis
# ============================================================

def contact_residue_set(receptor_atoms, ligand_coords, cutoff):
    contacts = set()

    for atom in receptor_atoms:
        for ligand_coord in ligand_coords:
            if distance(atom["coord"], ligand_coord) <= cutoff:
                contacts.add(residue_key(atom))
                break

    return contacts

# ============================================================
# Potential hydrogen-bond-compatible polar contact analysis
# ============================================================

SIDECHAIN_DONORS = {
    ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"),
    ("ASN", "ND2"), ("GLN", "NE2"),
    ("HIS", "ND1"), ("HIS", "NE2"),
    ("LYS", "NZ"),
    ("SER", "OG"), ("THR", "OG1"),
    ("TRP", "NE1"), ("TYR", "OH"),
    ("CYS", "SG")
}

SIDECHAIN_ACCEPTORS = {
    ("ASP", "OD1"), ("ASP", "OD2"),
    ("GLU", "OE1"), ("GLU", "OE2"),
    ("ASN", "OD1"), ("GLN", "OE1"),
    ("HIS", "ND1"), ("HIS", "NE2"),
    ("SER", "OG"), ("THR", "OG1"),
    ("TYR", "OH"), ("CYS", "SG"),
    ("MET", "SD")
}

def receptor_polar_roles(atom):
    resn = atom["resn"]
    atom_name = atom["atom_name"]

    donor = False
    acceptor = False

    # Backbone groups
    if atom_name == "N" and resn != "PRO":
        donor = True

    if atom_name == "O":
        acceptor = True

    # Side-chain groups
    if (resn, atom_name) in SIDECHAIN_DONORS:
        donor = True

    if (resn, atom_name) in SIDECHAIN_ACCEPTORS:
        acceptor = True

    return donor, acceptor

def potential_polar_contact_set(receptor_atoms, ligand_mol, cutoff):
    """
    Returns receptor-atom/direction level potential polar contacts.
    These are distance-based donor/acceptor-compatible contacts, not
    trajectory-derived hydrogen bond occupancies.
    """
    ligand_coords = molecule_coordinates(ligand_mol)

    ligand_donors = ligand_feature_atom_ids(ligand_mol, "Donor")
    ligand_acceptors = ligand_feature_atom_ids(ligand_mol, "Acceptor")

    contacts = set()

    for atom in receptor_atoms:

        receptor_is_donor, receptor_is_acceptor = receptor_polar_roles(atom)

        atom_id = (
            f"{atom['chain']}:{atom['resi']}:{atom['resn']}:{atom['atom_name']}"
        )

        if receptor_is_donor:
            for ligand_index in ligand_acceptors:
                if distance(atom["coord"], ligand_coords[ligand_index]) <= cutoff:
                    contacts.add(f"{atom_id}:PROT_DONOR_TO_QUE_ACCEPTOR")
                    break

        if receptor_is_acceptor:
            for ligand_index in ligand_donors:
                if distance(atom["coord"], ligand_coords[ligand_index]) <= cutoff:
                    contacts.add(f"{atom_id}:PROT_ACCEPTOR_FROM_QUE_DONOR")
                    break

    return contacts, len(ligand_donors), len(ligand_acceptors)

# ============================================================
# Read formal docking table
# ============================================================

with open(DOCKING_TABLE, "r", encoding="utf-8-sig", newline="") as handle:
    docking_rows = list(csv.DictReader(handle))

completed_rows = [
    row for row in docking_rows
    if row.get("docking_status") == "completed"
]

print("")
print("Completed QUE docking records found:", len(completed_rows))

if len(completed_rows) != 30:
    raise RuntimeError(
        f"Expected 30 completed QUE docking records, found {len(completed_rows)}."
    )

results = []

# ============================================================
# Analyze each QUE candidate
# ============================================================

for row in completed_rows:

    receptor_id = row["receptor_id"]

    receptor_pdb = RECEPTOR_DIR / f"{receptor_id}_aligned.pdb"
    reference_pdb = REFERENCE_LIGAND_DIR / f"{receptor_id}_QUE_reference.pdb"
    docked_sdf = DOCKING_SDF_DIR / f"{receptor_id}_QUE_out.sdf"

    try:
        receptor_atoms = parse_receptor_atoms(receptor_pdb)
        reference_ligand = build_reference_ligand(reference_pdb)
        docked_ligand, pose_count = read_docked_mode1(docked_sdf)

        reference_coords = molecule_coordinates(reference_ligand)
        docked_coords = molecule_coordinates(docked_ligand)

        in_place_rmsd = rdMolAlign.CalcRMS(
            docked_ligand,
            reference_ligand,
            maxMatches=100000
        )

        center_shift = centroid_shift(reference_coords, docked_coords)

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

        contact_recovery = percentage(
            len(recovered_contacts),
            len(reference_contacts)
        )

        new_contact_percent = percentage(
            len(new_contacts),
            len(docked_contacts)
        )

        contact_union = reference_contacts | docked_contacts

        contact_jaccard = (
            len(recovered_contacts) / len(contact_union)
            if len(contact_union) > 0 else ""
        )

        reference_polar, ligand_donor_count, ligand_acceptor_count = (
            potential_polar_contact_set(
                receptor_atoms,
                reference_ligand,
                POLAR_CONTACT_CUTOFF_A
            )
        )

        docked_polar, _, _ = potential_polar_contact_set(
            receptor_atoms,
            docked_ligand,
            POLAR_CONTACT_CUTOFF_A
        )

        recovered_polar = reference_polar & docked_polar
        lost_polar = reference_polar - docked_polar
        new_polar = docked_polar - reference_polar

        polar_recovery = percentage(
            len(recovered_polar),
            len(reference_polar)
        )

        new_polar_percent = percentage(
            len(new_polar),
            len(docked_polar)
        )

        polar_union = reference_polar | docked_polar

        polar_jaccard = (
            len(recovered_polar) / len(polar_union)
            if len(polar_union) > 0 else ""
        )

        affinity = float(row["best_affinity_kcal_mol"])
        pocket_rmsd = float(row["alignment_rmsd_A"])
        pocket_risk = row["pocket_alignment_risk"]

        polar_pass_A = (
            len(reference_polar) == 0 or
            (polar_recovery != "" and polar_recovery >= 50.0)
        )

        polar_pass_B = (
            len(reference_polar) == 0 or
            (polar_recovery != "" and polar_recovery >= 25.0)
        )

        if (
            in_place_rmsd <= 2.0
            and center_shift <= 1.0
            and contact_recovery != ""
            and contact_recovery >= 75.0
            and new_contact_percent != ""
            and new_contact_percent <= 30.0
            and polar_pass_A
            and pocket_risk == "Acceptable_pocket_alignment"
        ):
            tier = "Tier_A_connectivity_and_polar_supported_pose"

        elif (
            in_place_rmsd <= 3.0
            and center_shift <= 2.0
            and contact_recovery != ""
            and contact_recovery >= 50.0
            and new_contact_percent != ""
            and new_contact_percent <= 40.0
            and polar_pass_B
        ):
            tier = "Tier_B_partially_supported_pose"

        else:
            tier = "Tier_C_low_priority_pose"

        results.append({
            "receptor_id": receptor_id,
            "analysis_status": "success",
            "vina_affinity_kcal_mol": round(affinity, 4),
            "pocket_alignment_rmsd_A": round(pocket_rmsd, 4),
            "pocket_alignment_risk": pocket_risk,
            "poses_in_sdf": pose_count,
            "connectivity_aware_in_place_rmsd_A": round(float(in_place_rmsd), 4),
            "centroid_shift_A": round(float(center_shift), 4),
            "contact_cutoff_A": CONTACT_CUTOFF_A,
            "reference_contact_residue_count": len(reference_contacts),
            "docked_contact_residue_count": len(docked_contacts),
            "recovered_contact_residue_count": len(recovered_contacts),
            "lost_contact_residue_count": len(lost_contacts),
            "new_contact_residue_count": len(new_contacts),
            "contact_recovery_percent": round(contact_recovery, 4) if contact_recovery != "" else "",
            "new_contact_percent_of_docked": round(new_contact_percent, 4) if new_contact_percent != "" else "",
            "contact_jaccard": round(contact_jaccard, 4) if contact_jaccard != "" else "",
            "polar_contact_cutoff_A": POLAR_CONTACT_CUTOFF_A,
            "QUE_donor_atom_count": ligand_donor_count,
            "QUE_acceptor_atom_count": ligand_acceptor_count,
            "reference_polar_contact_count": len(reference_polar),
            "docked_polar_contact_count": len(docked_polar),
            "recovered_polar_contact_count": len(recovered_polar),
            "lost_polar_contact_count": len(lost_polar),
            "new_polar_contact_count": len(new_polar),
            "polar_contact_recovery_percent": round(polar_recovery, 4) if polar_recovery != "" else "",
            "new_polar_contact_percent_of_docked": round(new_polar_percent, 4) if new_polar_percent != "" else "",
            "polar_contact_jaccard": round(polar_jaccard, 4) if polar_jaccard != "" else "",
            "reference_contact_residues": format_items(reference_contacts),
            "docked_contact_residues": format_items(docked_contacts),
            "recovered_contact_residues": format_items(recovered_contacts),
            "new_contact_residues": format_items(new_contacts),
            "reference_polar_contacts": format_items(reference_polar),
            "docked_polar_contacts": format_items(docked_polar),
            "recovered_polar_contacts": format_items(recovered_polar),
            "new_polar_contacts": format_items(new_polar),
            "redocking_validation_tier": tier
        })

    except Exception as exc:
        results.append({
            "receptor_id": receptor_id,
            "analysis_status": "analysis_failed",
            "error": str(exc)
        })

# ============================================================
# Save complete metrics table
# ============================================================

metric_fields = [
    "receptor_id",
    "analysis_status",
    "vina_affinity_kcal_mol",
    "pocket_alignment_rmsd_A",
    "pocket_alignment_risk",
    "poses_in_sdf",
    "connectivity_aware_in_place_rmsd_A",
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
    "polar_contact_cutoff_A",
    "QUE_donor_atom_count",
    "QUE_acceptor_atom_count",
    "reference_polar_contact_count",
    "docked_polar_contact_count",
    "recovered_polar_contact_count",
    "lost_polar_contact_count",
    "new_polar_contact_count",
    "polar_contact_recovery_percent",
    "new_polar_contact_percent_of_docked",
    "polar_contact_jaccard",
    "reference_contact_residues",
    "docked_contact_residues",
    "recovered_contact_residues",
    "new_contact_residues",
    "reference_polar_contacts",
    "docked_polar_contacts",
    "recovered_polar_contacts",
    "new_polar_contacts",
    "redocking_validation_tier",
    "error"
]

with open(OUTPUT_METRICS, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=metric_fields)
    writer.writeheader()

    for result in results:
        writer.writerow({key: result.get(key, "") for key in metric_fields})

# ============================================================
# Priority ranking
# ============================================================

successful = [
    row for row in results
    if row.get("analysis_status") == "success"
]

tier_order = {
    "Tier_A_connectivity_and_polar_supported_pose": 0,
    "Tier_B_partially_supported_pose": 1,
    "Tier_C_low_priority_pose": 2
}

ranked = sorted(
    successful,
    key=lambda row: (
        tier_order.get(row["redocking_validation_tier"], 9),
        float(row["vina_affinity_kcal_mol"]),
        float(row["connectivity_aware_in_place_rmsd_A"]),
        -float(row["contact_recovery_percent"]) if row["contact_recovery_percent"] != "" else 999.0,
        -float(row["polar_contact_recovery_percent"]) if row["polar_contact_recovery_percent"] != "" else 999.0
    )
)

ranking_fields = [
    "receptor_id",
    "redocking_validation_tier",
    "vina_affinity_kcal_mol",
    "connectivity_aware_in_place_rmsd_A",
    "centroid_shift_A",
    "contact_recovery_percent",
    "new_contact_percent_of_docked",
    "polar_contact_recovery_percent",
    "new_polar_contact_percent_of_docked",
    "pocket_alignment_rmsd_A",
    "pocket_alignment_risk"
]

with open(OUTPUT_RANKING, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=ranking_fields)
    writer.writeheader()

    for result in ranked:
        writer.writerow({key: result.get(key, "") for key in ranking_fields})

# ============================================================
# Console output
# ============================================================

print("")
print("========== QUE POST-DOCKING ANALYSIS STATUS ==========")

status_counts = {}
for result in results:
    status = result.get("analysis_status", "")
    status_counts[status] = status_counts.get(status, 0) + 1

for status, count in status_counts.items():
    print(f"{status}: {count}")

print("")
print("========== QUE REDOCKING VALIDATION TIERS ==========")

tier_counts = {}
for result in successful:
    tier = result["redocking_validation_tier"]
    tier_counts[tier] = tier_counts.get(tier, 0) + 1

for tier, count in tier_counts.items():
    print(f"{tier}: {count}")

print("")
print("========== TOP 10 BY VINA AFFINITY WITH POSE METRICS ==========")

by_affinity = sorted(
    successful,
    key=lambda row: float(row["vina_affinity_kcal_mol"])
)

for rank, row in enumerate(by_affinity[:10], start=1):
    print(
        rank,
        row["receptor_id"],
        "| affinity:", row["vina_affinity_kcal_mol"],
        "| RMSD:", row["connectivity_aware_in_place_rmsd_A"],
        "| centroid:", row["centroid_shift_A"],
        "| contact recovery:", row["contact_recovery_percent"],
        "| polar recovery:", row["polar_contact_recovery_percent"],
        "| tier:", row["redocking_validation_tier"]
    )

print("")
print("========== TOP 15 BY VALIDATION PRIORITY ==========")

for rank, row in enumerate(ranked[:15], start=1):
    print(
        rank,
        row["receptor_id"],
        "| tier:", row["redocking_validation_tier"],
        "| affinity:", row["vina_affinity_kcal_mol"],
        "| RMSD:", row["connectivity_aware_in_place_rmsd_A"],
        "| centroid:", row["centroid_shift_A"],
        "| contact recovery:", row["contact_recovery_percent"],
        "| polar recovery:", row["polar_contact_recovery_percent"]
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
print("Complete metrics saved to:", OUTPUT_METRICS)
print("Priority ranking saved to:", OUTPUT_RANKING)
