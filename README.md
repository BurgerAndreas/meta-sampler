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

pip install pyscf numpy==1.24.4 plotly kaleido scipy matplotlib seaborn black tqdm joblib einops ipykernel toml hydra-core omegaconf nbformat

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
```

### Fit sampler 
```bash
python scripts/train_sampler.py --load_model trained_models/96_node_3_layer.pth --batch_size 4
```

## Info

OpenMM
https://openmm.org/

alanine_dipeptide.pdb
https://github.com/choderalab/YankTools/blob/master/testsystems/data/alanine-dipeptide-gbsa/alanine-dipeptide.pdb

amber99sb.xml (force field)
https://github.com/lpwgroup/tinker-openmm/blob/master/wrappers/python/simtk/openmm/app/data/amber99sb.xml