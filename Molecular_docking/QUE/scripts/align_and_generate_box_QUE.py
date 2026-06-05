from pathlib import Path
import csv
from pymol import cmd

# ============================================================
# Paths and settings
# ============================================================

ROOT = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\6_AutoDock_Vina\QUE")

MAPPING_FILE = ROOT / "QC" / "receptor_reference_mapping_QUE_corrected.csv"

ALIGNED_DIR = ROOT / "Aligned_Receptor_PDB"
CONFIG_DIR = ROOT / "box_parameters" / "configs"
REFERENCE_LIGAND_DIR = ROOT / "box_parameters" / "reference_ligands"
QC_DIR = ROOT / "QC"

for folder in [ALIGNED_DIR, CONFIG_DIR, REFERENCE_LIGAND_DIR, QC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

LIGAND_RESN = "QUE"

# QUE / quercetin is larger and more polar than 9IR.
# For this buried-pocket docking round, use a uniform 24 Å cubic box.
BOX_SIZE = 24.0

CONTACT_RADIUS = 5.0

results = []

# ============================================================
# Read corrected QUE mapping table
# ============================================================

if not MAPPING_FILE.exists():
    raise FileNotFoundError(f"Mapping file not found: {MAPPING_FILE}")

with open(MAPPING_FILE, "r", encoding="utf-8-sig", newline="") as handle:
    mapping_rows = list(csv.DictReader(handle))

print(f"QUE receptor-reference pairs detected: {len(mapping_rows)}")

if len(mapping_rows) != 32:
    raise RuntimeError(
        f"Expected 32 QUE receptor-reference pairs, but found {len(mapping_rows)}."
    )

# ============================================================
# Process each receptor-reference pair
# ============================================================

for row in mapping_rows:

    receptor_id = row["receptor_id"]
    backbone_id = row["backbone_id"]

    receptor_file = Path(row["receptor_file"])
    reference_file = Path(row["reference_file"])

    if not receptor_file.exists():
        results.append({
            "receptor_id": receptor_id,
            "backbone_id": backbone_id,
            "status": "missing_receptor_file"
        })
        print(f"[FAILED] Missing receptor: {receptor_id}")
        continue

    if not reference_file.exists():
        results.append({
            "receptor_id": receptor_id,
            "backbone_id": backbone_id,
            "status": "missing_reference_file"
        })
        print(f"[FAILED] Missing reference: {receptor_id}")
        continue

    cmd.reinitialize()

    cmd.load(str(reference_file), "ref")
    cmd.load(str(receptor_file), "rec")

    # ========================================================
    # Identify QUE in the RFdiffusion3 reference complex
    # ========================================================

    ligand_selection = f"ref and resn {LIGAND_RESN}"

    if cmd.count_atoms(ligand_selection) == 0:
        ligand_selection = "ref and organic and not polymer.protein"

    ligand_atom_count = cmd.count_atoms(ligand_selection)

    if ligand_atom_count == 0:
        results.append({
            "receptor_id": receptor_id,
            "backbone_id": backbone_id,
            "reference_file": str(reference_file),
            "status": "ligand_not_found"
        })
        print(f"[FAILED] QUE not found: {receptor_id}")
        continue

    # ========================================================
    # Define the QUE pocket correctly:
    # first collect full residues with any atom within 5 Å of QUE,
    # then use the C-alpha atoms of those residues for local fitting.
    # ========================================================

    pocket_ca_selection = (
        f"(byres ((ref and polymer.protein) within {CONTACT_RADIUS} of "
        f"({ligand_selection}))) "
        f"and ref and polymer.protein and name CA"
    )

    pocket_atoms = cmd.get_model(pocket_ca_selection).atom

    fit_pairs = []

    for atom in pocket_atoms:

        target_sel = (
            f"ref and polymer.protein and chain {atom.chain} "
            f"and resi {atom.resi} and name CA"
        )

        mobile_sel = (
            f"rec and polymer.protein and chain {atom.chain} "
            f"and resi {atom.resi} and name CA"
        )

        # Fallback if RF3 changed chain naming but retained residue numbering.
        if cmd.count_atoms(mobile_sel) != 1:
            mobile_sel = (
                f"rec and polymer.protein and resi {atom.resi} and name CA"
            )

        if (
            cmd.count_atoms(target_sel) == 1
            and cmd.count_atoms(mobile_sel) == 1
        ):
            fit_pairs.extend([mobile_sel, target_sel])

    pocket_ca_count = len(fit_pairs) // 2

    # ========================================================
    # Align receptor to RFdiffusion3 reference pocket
    # ========================================================

    if pocket_ca_count >= 3:

        alignment_method = "pocket_pair_fit"
        alignment_rmsd = cmd.pair_fit(*fit_pairs)

    else:

        # Keep the file for diagnosis, but structures using this fallback
        # should not enter docking until manually checked.
        alignment_method = "whole_protein_super"

        fit_result = cmd.super(
            "rec and polymer.protein and name CA",
            "ref and polymer.protein and name CA"
        )

        alignment_rmsd = fit_result[0]

    # ========================================================
    # Save aligned receptor and reference ligand
    # ========================================================

    aligned_pdb = ALIGNED_DIR / f"{receptor_id}_aligned.pdb"
    reference_ligand_pdb = (
        REFERENCE_LIGAND_DIR / f"{receptor_id}_QUE_reference.pdb"
    )
    config_file = CONFIG_DIR / f"{receptor_id}.box.txt"

    cmd.save(str(aligned_pdb), "rec and polymer.protein")
    cmd.save(str(reference_ligand_pdb), ligand_selection)

    # ========================================================
    # Generate docking box centered on reference QUE
    # ========================================================

    coordinates = cmd.get_coords(ligand_selection)

    center_x = sum(coord[0] for coord in coordinates) / len(coordinates)
    center_y = sum(coord[1] for coord in coordinates) / len(coordinates)
    center_z = sum(coord[2] for coord in coordinates) / len(coordinates)

    span_x = max(coord[0] for coord in coordinates) - min(coord[0] for coord in coordinates)
    span_y = max(coord[1] for coord in coordinates) - min(coord[1] for coord in coordinates)
    span_z = max(coord[2] for coord in coordinates) - min(coord[2] for coord in coordinates)

    config_file.write_text(
        f"center_x = {center_x:.3f}\n"
        f"center_y = {center_y:.3f}\n"
        f"center_z = {center_z:.3f}\n\n"
        f"size_x = {BOX_SIZE:.3f}\n"
        f"size_y = {BOX_SIZE:.3f}\n"
        f"size_z = {BOX_SIZE:.3f}\n",
        encoding="utf-8"
    )

    results.append({
        "receptor_id": receptor_id,
        "backbone_id": backbone_id,
        "receptor_file": str(receptor_file),
        "reference_file": str(reference_file),
        "aligned_receptor_pdb": str(aligned_pdb),
        "reference_ligand_pdb": str(reference_ligand_pdb),
        "config_file": str(config_file),
        "alignment_method": alignment_method,
        "pocket_ca_count": pocket_ca_count,
        "alignment_rmsd_A": round(float(alignment_rmsd), 4),
        "ligand_atom_count": ligand_atom_count,
        "ligand_span_x_A": round(float(span_x), 3),
        "ligand_span_y_A": round(float(span_y), 3),
        "ligand_span_z_A": round(float(span_z), 3),
        "center_x": round(float(center_x), 3),
        "center_y": round(float(center_y), 3),
        "center_z": round(float(center_z), 3),
        "box_size_A": BOX_SIZE,
        "status": "success"
    })

    print(
        f"[SUCCESS] {receptor_id} | "
        f"QUE_atoms={ligand_atom_count} | "
        f"pocket_CA={pocket_ca_count} | "
        f"RMSD={alignment_rmsd:.4f} Å | "
        f"box={BOX_SIZE:.1f} Å"
    )

# ============================================================
# Write manifest
# ============================================================

manifest_file = QC_DIR / "box_manifest_QUE.csv"

fieldnames = [
    "receptor_id",
    "backbone_id",
    "receptor_file",
    "reference_file",
    "aligned_receptor_pdb",
    "reference_ligand_pdb",
    "config_file",
    "alignment_method",
    "pocket_ca_count",
    "alignment_rmsd_A",
    "ligand_atom_count",
    "ligand_span_x_A",
    "ligand_span_y_A",
    "ligand_span_z_A",
    "center_x",
    "center_y",
    "center_z",
    "box_size_A",
    "status"
]

with open(manifest_file, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()

    for result in results:
        writer.writerow({key: result.get(key, "") for key in fieldnames})

success_count = sum(row.get("status") == "success" for row in results)
fallback_count = sum(
    row.get("alignment_method") == "whole_protein_super"
    for row in results
)
failed_count = len(mapping_rows) - success_count

print("")
print("========================================")
print("QUE alignment and box generation finished")
print("========================================")
print(f"Processed receptor count: {len(mapping_rows)}")
print(f"Success count: {success_count}")
print(f"Failed count: {failed_count}")
print(f"Whole protein fallback count: {fallback_count}")
print(f"Manifest file: {manifest_file}")