from pathlib import Path
import csv
from rdkit import Chem

# ============================================================
# Paths
# ============================================================

ROOT = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\AutoDock_Vina\QUE")

RECEPTOR_DIR = ROOT / "Aligned_Receptor_PDB"
DOCKED_SDF_DIR = ROOT / "Docking_30_Formal" / "outputs_sdf"
DOCKING_QC = ROOT / "Docking_30_Formal" / "QC" / "vina_formal_docking_status_and_affinity_QUE.csv"

OUT_ROOT = ROOT / "All_Complexes"
COMPLEX_DIR = OUT_ROOT / "complex_pdb"
LIGAND_PDB_DIR = OUT_ROOT / "ligand_pdb"
LIGAND_SDF_DIR = OUT_ROOT / "ligand_sdf_mode1"
QC_DIR = OUT_ROOT / "QC"

for folder in [OUT_ROOT, COMPLEX_DIR, LIGAND_PDB_DIR, LIGAND_SDF_DIR, QC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ============================================================
# Read docking summary
# ============================================================

if not DOCKING_QC.exists():
    raise FileNotFoundError(f"Docking summary not found: {DOCKING_QC}")

with open(DOCKING_QC, "r", encoding="utf-8-sig", newline="") as handle:
    docking_rows = list(csv.DictReader(handle))

completed = [
    row for row in docking_rows
    if row.get("docking_status") == "completed"
]

if len(completed) != 30:
    raise RuntimeError(
        f"Expected 30 completed QUE docking records, found {len(completed)}."
    )

# ============================================================
# Helper functions
# ============================================================

def read_first_pose_from_sdf(sdf_file):
    supplier = Chem.SDMolSupplier(str(sdf_file), removeHs=False)
    poses = [mol for mol in supplier if mol is not None]

    if not poses:
        raise RuntimeError(f"No readable pose found in: {sdf_file}")

    return poses[0], len(poses)

def ligand_to_pdb_lines(mol):
    pdb_block = Chem.MolToPDBBlock(mol)
    ligand_lines = []

    for line in pdb_block.splitlines():

        if line.startswith(("HETATM", "ATOM  ")):

            # Rename QUE as LIG, chain B, residue 1.
            # This avoids problems in downstream visualization/MD tools
            # caused by residue names beginning with numbers or custom IDs.
            new_line = (
                "HETATM" +
                line[6:17] +
                "LIG" +
                " B" +
                f"{1:4d}" +
                line[26:]
            )

            ligand_lines.append(new_line)

        elif line.startswith("CONECT"):
            # Connectivity is preserved in the mode1 SDF.
            # The complex PDB is mainly for visualization and MD structure upload.
            continue

    ligand_lines.append("END")
    return ligand_lines

def read_clean_protein_lines(pdb_file):
    protein_lines = []

    with open(pdb_file, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = line[:6].strip()

            # Remove terminal and connectivity records before merging ligand.
            if record in {"END", "CONECT", "MASTER"}:
                continue

            protein_lines.append(line.rstrip("\n"))

    return protein_lines

# ============================================================
# Generate all complexes
# ============================================================

records = []

for row in completed:

    receptor_id = row["receptor_id"]

    receptor_pdb = RECEPTOR_DIR / f"{receptor_id}_aligned.pdb"
    docked_sdf = DOCKED_SDF_DIR / f"{receptor_id}_QUE_out.sdf"

    mode1_sdf = LIGAND_SDF_DIR / f"{receptor_id}_QUE_mode1_for_CHARMM.sdf"
    ligand_pdb = LIGAND_PDB_DIR / f"{receptor_id}_QUE_mode1_ligand.pdb"
    complex_pdb = COMPLEX_DIR / f"{receptor_id}_QUE_mode1_complex.pdb"

    status = "success"
    error = ""

    try:
        if not receptor_pdb.exists():
            raise FileNotFoundError(f"Missing receptor PDB: {receptor_pdb}")

        if not docked_sdf.exists():
            raise FileNotFoundError(f"Missing docked SDF: {docked_sdf}")

        mode1, pose_count = read_first_pose_from_sdf(docked_sdf)

        heavy_atoms = sum(
            1 for atom in mode1.GetAtoms()
            if atom.GetAtomicNum() > 1
        )

        if heavy_atoms != 22:
            raise RuntimeError(
                f"QUE heavy atom count should be 22, found {heavy_atoms}"
            )

        # Add traceability metadata to mode1 SDF
        mode1.SetProp("_Name", f"{receptor_id}_QUE_mode1")
        mode1.SetProp("Ligand_ID", "QUE")
        mode1.SetProp("Ligand_description", "Quercetin")
        mode1.SetProp("Docking_pose", "AutoDock_Vina_mode_1")
        mode1.SetProp("Source_receptor_id", receptor_id)

        if "best_affinity_kcal_mol" in row:
            mode1.SetProp("Vina_affinity_kcal_mol", row["best_affinity_kcal_mol"])

        if "alignment_rmsd_A" in row:
            mode1.SetProp("Pocket_alignment_RMSD_A", row["alignment_rmsd_A"])

        if "pocket_alignment_risk" in row:
            mode1.SetProp("Pocket_alignment_risk", row["pocket_alignment_risk"])

        writer = Chem.SDWriter(str(mode1_sdf))
        writer.write(mode1)
        writer.close()

        ligand_lines = ligand_to_pdb_lines(mode1)

        ligand_pdb.write_text(
            "\n".join(ligand_lines) + "\n",
            encoding="ascii"
        )

        protein_lines = read_clean_protein_lines(receptor_pdb)

        complex_lines = protein_lines + ["TER"] + ligand_lines

        complex_pdb.write_text(
            "\n".join(complex_lines) + "\n",
            encoding="ascii"
        )

    except Exception as exc:
        status = "failed"
        error = str(exc)
        pose_count = ""
        heavy_atoms = ""

    records.append({
        "receptor_id": receptor_id,
        "status": status,
        "vina_affinity_kcal_mol": row.get("best_affinity_kcal_mol", ""),
        "pocket_alignment_rmsd_A": row.get("alignment_rmsd_A", ""),
        "pocket_alignment_risk": row.get("pocket_alignment_risk", ""),
        "docking_sdf": str(docked_sdf),
        "receptor_pdb": str(receptor_pdb),
        "mode1_sdf": str(mode1_sdf),
        "ligand_pdb": str(ligand_pdb),
        "complex_pdb": str(complex_pdb),
        "poses_in_original_sdf": pose_count,
        "QUE_heavy_atom_count": heavy_atoms,
        "error": error
    })

    print(f"[{status.upper()}] {receptor_id}")

# ============================================================
# Save manifest
# ============================================================

manifest = QC_DIR / "all_QUE_complex_generation_manifest.csv"

fieldnames = [
    "receptor_id",
    "status",
    "vina_affinity_kcal_mol",
    "pocket_alignment_rmsd_A",
    "pocket_alignment_risk",
    "docking_sdf",
    "receptor_pdb",
    "mode1_sdf",
    "ligand_pdb",
    "complex_pdb",
    "poses_in_original_sdf",
    "QUE_heavy_atom_count",
    "error"
]

with open(manifest, "w", newline="", encoding="utf-8-sig") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

success_count = sum(row["status"] == "success" for row in records)
failed_count = sum(row["status"] != "success" for row in records)

print("")
print("==============================================")
print("QUE protein-ligand complex generation finished")
print("==============================================")
print(f"Input completed docking records: {len(completed)}")
print(f"Successful complexes: {success_count}")
print(f"Failed complexes: {failed_count}")
print("")
print(f"Complex PDB folder: {COMPLEX_DIR}")
print(f"Mode1 SDF folder: {LIGAND_SDF_DIR}")
print(f"Ligand PDB folder: {LIGAND_PDB_DIR}")
print(f"Manifest: {manifest}")

