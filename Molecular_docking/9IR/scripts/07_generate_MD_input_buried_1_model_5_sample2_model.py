from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors

# ============================================================
# Target candidate and paths
# ============================================================

ROOT = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\6_AutoDock_Vina\9IR")

CANDIDATE = "buried_1_model_5_sample2_model"

RECEPTOR_PDB = ROOT / "Aligned_Receptor_PDB" / f"{CANDIDATE}_aligned.pdb"

DOCKING_SDF = (
    ROOT / "Docking_48_Formal" / "outputs_sdf" /
    f"{CANDIDATE}_9IR_out.sdf"
)

MD_DIR = ROOT / "MD_Input" / CANDIDATE
MD_DIR.mkdir(parents=True, exist_ok=True)

MODE1_SDF = MD_DIR / f"{CANDIDATE}_9IR_mode1_for_CHARMM.sdf"
MODE1_LIGAND_PDB = MD_DIR / f"{CANDIDATE}_9IR_mode1_ligand.pdb"
COMPLEX_PDB = MD_DIR / f"{CANDIDATE}_9IR_mode1_complex_for_MD.pdb"
MANIFEST = MD_DIR / f"{CANDIDATE}_MD_input_manifest.txt"

# Confirmed screening metrics
VINA_AFFINITY = -6.964
IN_PLACE_RMSD = 0.7057
CENTROID_SHIFT = 0.0495
CONTACT_RECOVERY = 93.3333
NEW_CONTACT_PERCENT = 0.0000

# ============================================================
# Check required input files
# ============================================================

if not RECEPTOR_PDB.exists():
    raise FileNotFoundError(f"Aligned receptor not found: {RECEPTOR_PDB}")

if not DOCKING_SDF.exists():
    raise FileNotFoundError(f"Docking SDF not found: {DOCKING_SDF}")

# ============================================================
# Read docking SDF and extract Vina mode 1
# ============================================================

supplier = Chem.SDMolSupplier(str(DOCKING_SDF), removeHs=False)
poses = [mol for mol in supplier if mol is not None]

if not poses:
    raise RuntimeError(f"No readable docking pose found in: {DOCKING_SDF}")

mode1 = poses[0]

if mode1.GetNumConformers() != 1:
    raise RuntimeError("Mode 1 ligand does not contain exactly one conformer.")

heavy_atoms = sum(1 for atom in mode1.GetAtoms() if atom.GetAtomicNum() > 1)

if heavy_atoms != 10:
    raise RuntimeError(
        f"Unexpected 9IR heavy atom count: {heavy_atoms}; expected 10."
    )

# Preserve traceability information in SDF properties
mode1.SetProp("_Name", f"{CANDIDATE}_9IR_mode1")
mode1.SetProp("Ligand_ID", "9IR")
mode1.SetProp("Ligand_description", "(4R)-limonene")
mode1.SetProp("Docking_pose", "AutoDock_Vina_mode_1")
mode1.SetProp("Vina_affinity_kcal_mol", str(VINA_AFFINITY))
mode1.SetProp("Connectivity_aware_in_place_RMSD_A", str(IN_PLACE_RMSD))
mode1.SetProp("Centroid_shift_A", str(CENTROID_SHIFT))
mode1.SetProp("Contact_recovery_percent", str(CONTACT_RECOVERY))
mode1.SetProp("New_contact_percent_of_docked", str(NEW_CONTACT_PERCENT))

writer = Chem.SDWriter(str(MODE1_SDF))
writer.write(mode1)
writer.close()

# ============================================================
# Write mode 1 ligand PDB
# ============================================================

ligand_pdb_block = Chem.MolToPDBBlock(mode1)

# Normalize ligand residue name and chain for the MD complex.
# LIG is used instead of 9IR to avoid possible downstream issues with
# residue names beginning with a digit.
ligand_lines = []

for line in ligand_pdb_block.splitlines():
    if line.startswith(("HETATM", "ATOM  ")):
        line = (
            "HETATM" +
            line[6:17] +
            "LIG" +
            " B" +
            f"{1:4d}" +
            line[26:]
        )
        ligand_lines.append(line)
    elif line.startswith("CONECT"):
        # Connectivity is preserved in the SDF supplied separately.
        # Do not insert CONECT records into the protein-ligand complex PDB.
        continue

ligand_lines.append("END")

MODE1_LIGAND_PDB.write_text(
    "\n".join(ligand_lines) + "\n",
    encoding="ascii"
)

# ============================================================
# Build protein–ligand complex PDB
# ============================================================

protein_lines = []

with open(RECEPTOR_PDB, "r", encoding="utf-8", errors="replace") as handle:
    for line in handle:
        record = line[:6].strip()

        # Preserve protein coordinates only.
        # Remove previous END/CONECT records before appending docked ligand.
        if record in {"END", "CONECT", "MASTER"}:
            continue

        protein_lines.append(line.rstrip("\n"))

complex_lines = protein_lines + ["TER"] + ligand_lines

COMPLEX_PDB.write_text(
    "\n".join(complex_lines) + "\n",
    encoding="ascii"
)

# ============================================================
# Write manifest
# ============================================================

manifest_text = f"""MD input preparation for 9IR candidate
===================================

Candidate ID:
{CANDIDATE}

Ligand:
9IR / (4R)-limonene

Protein coordinate source:
RoseTTAFold3 predicted receptor aligned to the corresponding RFdiffusion3 reference pocket.

Ligand coordinate source:
AutoDock Vina v1.2.7 docking output, mode 1 pose extracted from:
{DOCKING_SDF}

Files generated:
1. Ligand SDF for parameterization:
   {MODE1_SDF}

2. Ligand PDB for visual inspection:
   {MODE1_LIGAND_PDB}

3. Protein–ligand complex PDB for MD setup:
   {COMPLEX_PDB}

Selection metrics:
Vina affinity (kcal/mol): {VINA_AFFINITY}
Connectivity-aware in-place RMSD (A): {IN_PLACE_RMSD}
Centroid shift (A): {CENTROID_SHIFT}
Contact recovery (%): {CONTACT_RECOVERY}
New contact percentage (%): {NEW_CONTACT_PERCENT}

Important notes:
- The complex contains the RoseTTAFold3-derived protein and the Vina mode 1 docked 9IR pose.
- The RFdiffusion3 original complex was used only as a reference pocket/pose and is not used as the MD starting complex.
- In the complex PDB, the ligand residue is named LIG on chain B, residue 1.
- The accompanying SDF preserves the ligand chemical connectivity and docked coordinates for CHARMM-GUI ligand parameterization.
"""

MANIFEST.write_text(manifest_text, encoding="utf-8")

# ============================================================
# Final checks
# ============================================================

print("==============================================")
print("9IR MD input files generated successfully")
print("==============================================")
print(f"Candidate: {CANDIDATE}")
print(f"Docking poses found in original SDF: {len(poses)}")
print(f"Selected pose: mode 1")
print(f"Heavy atoms in selected ligand: {heavy_atoms}")
print(f"Total ligand atoms including H: {mode1.GetNumAtoms()}")
print(f"Molecular weight: {Descriptors.MolWt(mode1):.4f}")
print("")
print(f"Ligand SDF: {MODE1_SDF}")
print(f"Ligand PDB: {MODE1_LIGAND_PDB}")
print(f"Complex PDB: {COMPLEX_PDB}")
print(f"Manifest: {MANIFEST}")
