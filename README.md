# Data-free Diffusion Model for Sampling Transition Points

### Install
```bash
# get mamba package manager
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

mamba create -n painn python=3.10
mamba activate painn
pip uninstall painn -y
pip install -e .

pip install pyscf numpy==1.24.4 scikit-learn plotly kaleido scipy matplotlib seaborn black tqdm joblib einops ipykernel toml hydra-core omegaconf nbformat openmm 

# First uninstall any existing installations
pip uninstall torch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -y

# Install PyTorch for CUDA 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
# pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# Finally install torch-geometric
pip install torch-geometric

# alternative
# mamba install pytorch pytorch-cuda=12.6 -c pytorch -c nvidia
# mamba install pyg -c pyg

# for fast mace
# pip install cuequivariance-ops-torch-cu12
# git clone https://github.com/ACEsuit/mace
# pip install mace/.

# for slow mace
pip install mace-torch
```

### Fit sampler 
```bash
python scripts/train_sampler.py --load_model trained_models/96_node_3_layer.pth --batch_size 4
```

## Info

OpenMM
https://openmm.org/

amber99sb.xml (force field)
https://github.com/lpwgroup/tinker-openmm/blob/master/wrappers/python/simtk/openmm/app/data/amber99sb.xml

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

The backbone dihedral angles (φ and ψ) are defined by:
φ (phi): C(ACE)-N(ALA)-CA(ALA)-C(ALA) [atoms 5-7-9-15]
ψ (psi): N(ALA)-CA(ALA)-C(ALA)-N(NME) [atoms 7-9-15-17]
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