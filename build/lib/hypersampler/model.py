import torch
from torch import nn

# from PaiNN.model import PainnModel

"""
Batch in the dataset contains:
- num_atoms: number of atoms in the batch[B] # use
- elems: atom type Z [n_atoms_batch]. n_atoms_batch = sum(num_atoms) # use
- cell: cell [B, 3, 3] # use
- coord: atom positions [n_atoms_batch, 3] # predict
- pairs: pairs of atoms [n_pairs_batch, 2]. n_pairs_batch = sum(num_pairs) # recompute after prediction
- n_diff: difference in positions between pairs [n_pairs_batch, 3] # recompute after prediction
- num_pairs: number of pairs in the batch [B] # recompute after prediction
- energy: potential energy [B] # ignore
- forces: forces [n_atoms_batch, 3] # ignore
"""

class HyperSampler(nn.Module):
    """NN that learns to predicts extreme points in the potential energy surface.
    Acts as a hypernetwork for the PainnModel.
    The input is a set of atom types and the output are the atom coordinates.
    The PainnModel weights are used as additional input?
    """
    def __init__(self, n_painn_weights: int = -1, hidden_state_size: int = 128):
        super().__init__()
        num_embedding = 119  # Same as PaiNN
        self.hidden_state_size = hidden_state_size
        self.n_painn_weights = n_painn_weights
        self.atom_embedding = nn.Embedding(num_embedding, hidden_state_size)
        
        # Parameter processing layers (only used if use_painn_weights=True)
        if n_painn_weights > 0:
            self.param_encoder = nn.Sequential(
                nn.Linear(hidden_state_size, hidden_state_size),
                nn.SiLU(),
                nn.Linear(hidden_state_size, hidden_state_size)
            )
            
            # Modified MLP to include parameter information
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_state_size * 2, hidden_state_size * 2),  # *2 for concatenated features
                nn.SiLU(),
                nn.Linear(hidden_state_size * 2, hidden_state_size),
                nn.SiLU(),
                nn.Linear(hidden_state_size, 3)  # Output 3D coordinates
            )
        else:
            # Original MLP without parameter conditioning
            self.coord_mlp = nn.Sequential(
                nn.Linear(hidden_state_size, hidden_state_size * 2),
                nn.SiLU(),
                nn.Linear(hidden_state_size * 2, hidden_state_size),
                nn.SiLU(),
                nn.Linear(hidden_state_size, 3)  # Output 3D coordinates
            )

    def forward(self, input_dict: dict, params: dict = None):
        """
        Args:
            input_dict: dict with the following keys:
                - num_atoms: number of atoms in the batch[B] # use
                - elems: atom type Z [n_atoms_batch]. n_atoms_batch = sum(num_atoms) # use
                - cell: cell [B, 3, 3] # use
            params: model parameters (optional) (n_params)
        Returns:
            coord: predicted positions (n_atoms_batch, 3)
        """
        # Encode atom types using the same embedding as PaiNN
        node_scalar = self.atom_embedding(input_dict['elems'])
        
        if self.n_painn_weights > 0 and params is not None:
            # Process PaiNN parameters
            param_features = self.param_encoder(params['features'])
            # Expand param features to match batch size
            param_features = param_features.unsqueeze(0).expand(node_scalar.size(0), -1)
            # Concatenate with node features
            node_features = torch.cat([node_scalar, param_features], dim=-1)
        else:
            node_features = node_scalar
        
        # Predict coordinates for each atom
        coord = self.coord_mlp(node_features)
        
        # Scale coordinates to fit within the unit cell
        batch_idx = torch.repeat_interleave(
            torch.arange(len(input_dict['num_atoms']), device=coord.device),
            input_dict['num_atoms']
        )
        
        # Apply cell transformation to ensure atoms are within bounds
        coord = torch.sigmoid(coord)  # Scale to [0,1]
        coord = torch.bmm(
            coord.unsqueeze(1),
            input_dict['cell'][batch_idx]
        ).squeeze(1)
        
        return coord
