# Data-free Diffusion Model for Sampling Transition Points

### Install
```bash
# get mamba package manager
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

mamba create -n sampler python=3.13
mamba activate sampler

pip install numpy==1.24.4 scikit-learn plotly kaleido imageio scipy matplotlib seaborn black tqdm joblib einops ipykernel toml omegaconf nbformat openmm nglview py3Dmol hydra-core==1.*
```

Install PyTorch and PyTorch Geometric
```bash
# First uninstall any existing installations
pip uninstall torch torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -y

# Install PyTorch for CUDA 12.6
pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -U pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

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

Install Boltzmann Generator Utils
```bash
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

Install DEM sampler
```bash
pip install lightning==2.* torchmetrics==0.* hydra-core==1.* rich==13.* pre-commit==3.* pytest==7.* wandb hydra-optuna-sweeper hydra-colorlog rootutils normflows nflows einops torchsde torchdiffeq torchcfm 
pip install git+https://github.com/VincentStimper/resampled-base-flows.git git+https://github.com/jarridrb/fab-torch.git 

cd ..
git clone https://github.com/atong01/bgflow.git
cd bgflow
# Fix the SafeConfigParser issue
sed -i 's/configparser.SafeConfigParser()/configparser.ConfigParser()/g' versioneer.py
# Fix the readfp issue
sed -i 's/parser.readfp(f)/parser.read_file(f)/g' versioneer.py
pip install -e .
cd ..

cd DEM
pip install -e .
```

### Fit sampler 
```bash
```

