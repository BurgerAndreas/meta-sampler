# Alanine dipeptide

## Force fields
OpenMM
https://openmm.org/

amber99sb.xml (force field)
https://github.com/lpwgroup/tinker-openmm/blob/master/wrappers/python/simtk/openmm/app/data/amber99sb.xml

### Mace
https://github.com/ACEsuit/mace

MACE-OFF23, 10 elements covered, trained on SPICE v1, on DFT (wB97M+D3) data, designed for Organic Chemistry

recommended to use float64 for relaxations. We mostly use float32 at the moment.

## alanine dipeptide pdb files

alanine_dipeptide.pdb
https://github.com/choderalab/YankTools/blob/master/testsystems/data/alanine-dipeptide-gbsa/alanine-dipeptide.pdb

https://github.com/noegroup/bgflow/blob/fbba56fac3eb88f6825d2bd4f745ee75ae9715e1/tests/data/alanine-dipeptide-nowater.pdb

consists of 3 residues:
ACE (acetyl group) - N-terminal cap
ALA (alanine) - the central amino acid
3. NME (N-methylamide) - C-terminal cap

each line explained:
```
CRYST1: Crystal structure information (unit cell parameters)

ACE (Acetyl cap) - Residue 1:
1. HH31 - Hydrogen of methyl group
2. CH3  - Carbon of methyl group
3. HH32 - Hydrogen of methyl group
4. HH33 - Hydrogen of methyl group
5. C    - Carbonyl carbon
6. O    - Carbonyl oxygen

ALA (Alanine) - Residue 2:
7. N    - Backbone nitrogen
8. H    - Nitrogen-bound hydrogen
9. CA   - Alpha carbon
10. HA   - Alpha hydrogen
11. CB   - Beta carbon (alanine's methyl group)
12. HB1  - Beta hydrogen
13. HB2  - Beta hydrogen
14. HB3  - Beta hydrogen
15. C    - Carbonyl carbon
16. O    - Carbonyl oxygen

NME (N-methylamide cap) - Residue 3:
17. N    - Nitrogen
18. H    - Nitrogen-bound hydrogen
19. CH3  - Carbon of methyl group
20. HH31 - Hydrogen of methyl group
21. HH32 - Hydrogen of methyl group
22. HH33 - Hydrogen of methyl group
```
the units for atomic coordinates and unit cell dimensions are in ångströms (Å).

The backbone dihedral angles (φ and ψ) are defined by:
φ (phi): C(ACE)-N(ALA)-CA(ALA)-C(ALA) 
ψ (psi): N(ALA)-CA(ALA)-C(ALA)-N(NME) 
Each line contains:
ATOM: Record type
Number: Atom serial number (1-22)
Name: Atom name (HH31, CH3, etc.)
Residue: Three-letter code (ACE/ALA/NME)
Chain ID: X
Residue number: 1, 2, or 3
XYZ coordinates (in Angstroms)
Occupancy (1.00)
Temperature factor (0.00)



