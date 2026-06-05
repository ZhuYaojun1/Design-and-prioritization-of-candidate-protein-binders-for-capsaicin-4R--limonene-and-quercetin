from pathlib import Path
from rdkit import Chem

input_sdf = Path(r"C:\Users\Lenovo\Desktop\small_molecule_design\6_AutoDock_Vina\9IR\9IR_PDBQT\9IR_ideal.sdf")

supplier = Chem.SDMolSupplier(str(input_sdf), removeHs=False)
mol = next((m for m in supplier if m is not None), None)

if mol is None:
    raise RuntimeError("Failed to read 9IR_ideal.sdf")

heavy_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
hydrogen_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 1)
conformers = mol.GetNumConformers()
is_3d = mol.GetConformer().Is3D() if conformers else False

print("Heavy atoms:", heavy_atoms)
print("Explicit hydrogen atoms:", hydrogen_atoms)
print("Conformer count:", conformers)
print("Is 3D:", is_3d)
