After downloading the CIF file for a small molecule (e.g., from https://www.rcsb.org/ligand/9IR), modify the file using the provided templates to make it compatible with RFdiffusion3 input.
Template links:

https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/examples/sm_binder_design.json

https://github.com/RosettaCommons/foundry/blob/production/models/rfd3/docs/input_pdbs/IAI.pdb

Operations to perform in PyMOL:

Add a chain ID (e.g., L)

Change the residue number from 0 to 1 (or to a specific number, e.g., 392 in the example)

Change occupancy from 0.00 to 1.00

For RFdiffusion3 input, remove hydrogens and keep only heavy atoms.

PyMOL commands:

remove hydro
alter all, resn="9IR"
alter all, chain="B"
alter all, resi="392"
alter all, segi=""
alter all, q=1.00
alter all, b=0.00
alter all, type="HETATM"
sort

This procedure produces a PDB structure ready for RFdiffusion3 input.
