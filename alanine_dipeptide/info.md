# Alanine dipeptide

## Force fields
OpenMM
https://openmm.org/

amber99sb.xml (force field)
https://github.com/lpwgroup/tinker-openmm/blob/master/wrappers/python/simtk/openmm/app/data/amber99sb.xml

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

## Energy of alanine dipeptide

The energy of alanine dipeptide varies depending on its conformation, but there are several key points to consider:

1. The lowest energy conformations of alanine dipeptide typically have energies below 6 kcal/mol[^2].
2. The energy landscape of alanine dipeptide is often represented by a Ramachandran plot, which shows the energy as a function of the φ and ψ dihedral angles[^3].
3. The minimum potential energy of alanine dipeptide occurs when the backbone dihedral angles are positioned at Φ/Ψ = -80°/80°, which is denoted as a C conformation[^6].
4. The average thermal energy (kBT) of alanine dipeptide at 300 K is 2.493 kJ/mol, which is equivalent to about 0.596 kcal/mol[^5].

It's important to note that the energy of alanine dipeptide can vary based on factors such as the surrounding environment (e.g., vacuum vs. solvent) and the specific conformation it adopts. The energy landscape of alanine dipeptide is complex, with multiple local minima, making it a challenging system to study accurately[^2][^4].


[^1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC3913540/

[^2]: https://onlinelibrary.wiley.com/doi/am-pdf/10.1002/jcc.25589

[^3]: https://www.researchgate.net/figure/Protein-energy-and-dynamics-A-The-energy-of-alanine-dipeptide-in-the-learned-potential_fig3_354340003

[^4]: https://pubs.acs.org/doi/10.1021/jp100950w

[^5]: https://www.researchgate.net/figure/Free-energy-profile-of-alanine-dipeptide-as-a-function-of-the-dihedral-angles-Energies_fig2_266024359

[^6]: https://www.whxb.pku.edu.cn/EN/abstract/abstract28072.shtml

[^7]: https://pubs.acs.org/doi/10.1021/acs.jpcb.7b01130

