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

pip install pyscf numpy==1.24.4 scikit-learn plotly kaleido imageio scipy matplotlib seaborn black tqdm joblib einops ipykernel toml hydra-core omegaconf nbformat openmm nglview py3Dmol

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

# for fast mace (5-9x speedup)
pip uninstall mace-torch -y
# mace==0.3.10 was designed to work with cuequivariance==0.1.0
pip install cuequivariance==0.1.0 cuequivariance-torch cuequivariance-ops-torch-cu12
cd ..
git clone https://github.com/ACEsuit/mace
pip install mace/.
cd meta-sampler
```

```bash
# install boltzmann generator utils
pip install mdtraj==1.9.9 tables

cd ..
git clone git@github.com:noegroup/bgmol.git
cd bgmol
python setup.py install
cd ..

git clone git@github.com:noegroup/bgflow.git
cd bgflow
python setup.py install
cd ..

cd meta-sampler
```

### Fit sampler 
```bash
python scripts/train_sampler.py --load_model trained_models/96_node_3_layer.pth --batch_size 4
```

