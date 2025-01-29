import torch
from torch import nn
from torch.nn import MultiheadAttention
import time
import plotly.graph_objects as go
import matplotlib.pyplot as plt

"""
Batch in the dataset contains:
- num_atoms: number of atoms in the batch[B] # use
- elems: atom type Z [n_atoms_batch]. n_atoms_batch = sum(num_atoms) # use
- cell: cell [B*3, 3] # use
- coord: atom positions [n_atoms_batch, 3] # predict
- pairs: pairs of atoms [n_pairs_batch, 2]. n_pairs_batch = sum(num_pairs) # recompute after prediction
- n_diff: difference in positions between pairs [n_pairs_batch, 3] # recompute after prediction
- num_pairs: number of pairs in the batch [B] # recompute after prediction
- energy: potential energy [B] # ignore
- forces: forces [n_atoms_batch, 3] # ignore
"""

class GompertzActivation(torch.nn.Module):
    def __init__(self, a=5.0, b=0.0):
        super().__init__()
        self.a = torch.tensor(a)
        self.b = torch.tensor(b)

    def forward(self, x):
        return torch.exp(-torch.exp(-self.a * (x - self.b)))

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        return x + self.mlp(x)

class SamplerModel(nn.Module):
    """Neural network that predicts small perturbations to input coordinates to find 
    extreme points in the potential energy surface.
    """
    def __init__(self, hidden_state_size: int = 128):
        super().__init__()
        num_embedding = 119  # Same as PaiNN
        self.hidden_state_size = hidden_state_size
        
        # Atom type embedding
        self.atom_embedding = nn.Embedding(num_embedding, hidden_state_size)
        
        # Cell encoder
        self.cell_encoder = nn.Sequential(
            nn.Linear(3, hidden_state_size),
            nn.SiLU(),
            nn.Linear(hidden_state_size, hidden_state_size)
        )
        
        # Layer normalizations
        self.input_norm = nn.LayerNorm(hidden_state_size)
        self.feature_norm = nn.LayerNorm(hidden_state_size * 2 + 3)  # For combined features
        
        # Attention for atom interactions
        self.self_attention = MultiheadAttention(hidden_state_size, num_heads=4)
        
        # Offset predictor - initialized to predict values close to zero
        self.offset_predictor = nn.Sequential(
            nn.Linear(hidden_state_size * 2 + 3, hidden_state_size),
            nn.LayerNorm(hidden_state_size),
            nn.SiLU(),
            ResidualBlock(hidden_state_size),
            nn.LayerNorm(hidden_state_size),
            nn.Linear(hidden_state_size, 3),
            nn.Tanh()  # Constrain output to [-1, 1]
        )
        
        # Initialize last layer to output very small values
        self.offset_predictor[-2].weight.data.fill_(0.0)
        self.offset_predictor[-2].bias.data.fill_(0.0)

    def forward(self, num_atoms, elems, cell, coord):
        """
        Args:
            num_atoms: number of atoms in each molecule [B]
            elems: atom types [n_atoms_batch]
            cell: unit cell [B*3, 3]
            coord: input coordinates [n_atoms_batch, 3]
        Returns:
            perturbed coordinates [n_atoms_batch, 3]
        """
        # Encode atom types
        atom_features = self.atom_embedding(elems)
        atom_features = self.input_norm(atom_features)
        
        # Apply self-attention for atom interactions
        atom_features = self.self_attention(
            atom_features, atom_features, atom_features
        )[0]
        
        # Encode cell and expand to match number of atoms
        cell_features = self.cell_encoder(cell)
        batch_idx = torch.repeat_interleave(
            torch.arange(len(num_atoms), device=coord.device),
            repeats=num_atoms
        )
        cell_features = cell_features.repeat(len(num_atoms), 1)[batch_idx]
        
        # Combine all features
        combined_features = torch.cat([atom_features, cell_features, coord], dim=-1)
        combined_features = self.feature_norm(combined_features)
        
        # Predict small coordinate offsets
        offset = self.offset_predictor(combined_features)
        offset = offset * 0.01  # Scale down the offset predictions
        
        # Add offset to input coordinates
        perturbed_coord = coord + offset
        # return perturbed_coord
        
        # this is probably not necessary
        # Scale coordinates to fit within the unit cell
        prev_atoms = 0
        scaled_coords = []
        for b, n_atoms in enumerate(num_atoms):
            batch_coord = perturbed_coord[prev_atoms:prev_atoms + n_atoms]
            batch_cell = cell[b*3:(b+1)*3]
            
            # Clamp the coordinates to the unit cell
            # Maybe it's better to 'teleport' the atoms to the other side of the cell?
            scaled_batch_coord = torch.clamp(
                batch_coord, min=batch_cell.min(dim=1).values + 1e-6, 
                max=batch_cell.max(dim=1).values - 1e-6
            )
            
            # check if coords are inside the cell
            cellmax = batch_cell.max(dim=1).values
            inside_cell_max = torch.all(scaled_batch_coord < cellmax)
            assert inside_cell_max, f"inside cell max: \n{scaled_batch_coord[scaled_batch_coord >= cellmax]}"
            cellmin = batch_cell.min(dim=1).values
            inside_cell_min = torch.all(scaled_batch_coord > cellmin)
            assert inside_cell_min, f"inside cell min: {scaled_batch_coord[scaled_batch_coord <= cellmin]}"
            
            scaled_coords.append(scaled_batch_coord)
            prev_atoms += n_atoms
            

            # # 3d plot of cell with plotly
            # _coord = scaled_batch_coord.detach().cpu().numpy()
            # _cell = batch_cell.detach().cpu().numpy()
            # fig = go.Figure()
            # # Add atoms scatter
            # fig.add_trace(go.Scatter3d(
            #     x=_coord[:, 0], y=_coord[:, 1], z=_coord[:, 2],
            #     mode='markers',
            #     marker=dict(size=5)
            # ))
            # # Add cell lines
            # fig.add_trace(go.Scatter3d(
            #     x=[0, _cell[0, 0]], y=[0, _cell[1, 0]], z=[0, _cell[2, 0]],
            #     mode='lines', line=dict(color='black')
            # ))
            # fig.add_trace(go.Scatter3d(
            #     x=[0, _cell[0, 1]], y=[0, _cell[1, 1]], z=[0, _cell[2, 1]], 
            #     mode='lines', line=dict(color='black')
            # ))
            # fig.add_trace(go.Scatter3d(
            #     x=[0, _cell[0, 2]], y=[0, _cell[1, 2]], z=[0, _cell[2, 2]],
            #     mode='lines', line=dict(color='black')
            # ))
            # fname = f"coord_pred_{b}.png"
            # fig.write_image(fname)
            # print(f"Plotly plot saved to {fname}")
            
        # Concatenate all scaled coordinates
        return torch.cat(scaled_coords, dim=0)
        


def get_parameters(model: nn.Module):
    # return {k: v.detach().cpu().numpy() for k, v in model.named_parameters()}
    return torch.cat([p.flatten() for p in model.parameters()])