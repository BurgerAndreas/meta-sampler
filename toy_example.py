import torch
from tqdm import tqdm


# toy ML force field
class ToyMLFF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Input features: flattened positions (3*n_atoms)
        self.fc1 = torch.nn.Linear(9, 32)
        self.fc2 = torch.nn.Linear(32, 16)
        self.fc3 = torch.nn.Linear(16, 1)  # Output scalar energy

    def forward(self, x, z):
        """
        x: atomic positions (3 x n_atoms)
        z: atomic numbers (n_atoms)
        return: energy (scalar), forces per atom (3 x n_atoms)
        """
        # Compute energy
        batch_size = x.shape[0] if len(x.shape) > 2 else 1
        x_flat = x.reshape(batch_size, -1)  # Flatten positions

        x_flat.requires_grad_(True)  # Enable gradient computation

        h = torch.relu(self.fc1(x_flat))
        h = torch.relu(self.fc2(h))
        energy = self.fc3(h)  # Scalar energy

        # Compute forces as negative gradient of energy w.r.t. positions
        forces = -torch.autograd.grad(
            energy.sum(), x_flat, create_graph=True, retain_graph=True
        )[0]

        forces = forces.reshape(batch_size, -1, 3)  # Reshape to (batch, n_atoms, 3)

        return energy, forces

    def get_parameters(self):
        """Returns flattened parameters"""
        return torch.cat([p.flatten() for p in self.parameters()])


class MetaSampler(torch.nn.Module):
    """Predicts the atom positions x that minimize the energy E(x).
    Takes as input the model parameters and atom numbers z.
    Returns the predicted positions.
    """

    def __init__(self, n_params, n_atoms):
        super().__init__()
        self.n_params = n_params
        self.n_atoms = n_atoms

        # simple MLP to predict the positions
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(n_params + n_atoms, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 3 * n_atoms),
        )

    def forward(self, params, z):
        """
        params: model parameters (n_params)
        z: atomic numbers (n_atoms)
        return: predicted positions (3 x n_atoms)
        """
        params = params.reshape(1, -1)
        x = self.mlp(torch.cat([params, z], dim=-1))
        return x.reshape(-1, self.n_atoms, 3)


if __name__ == "__main__":
    # Example usage
    x = torch.randn(1, 3, 3)  # Batch size 1, 3 atoms, 3D positions
    z = torch.randint(1, 10, (1, 3))  # Atomic numbers

    model = ToyMLFF()
    energy, forces = model(x, z)

    print(f"energy: {energy.shape}")
    print(f"forces: {forces.shape}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")

    # meta sampler
    n_params = sum(p.numel() for p in model.parameters())
    n_atoms = x.shape[-1]
    meta_sampler = MetaSampler(n_params, n_atoms)
    x_pred = meta_sampler(model.get_parameters(), z)
    print(f"x_pred: {x_pred.shape}")

    # train meta sampler
    optimizer = torch.optim.Adam(meta_sampler.parameters(), lr=1e-3)
    for _ in tqdm(range(100)):
        optimizer.zero_grad()
        x_pred = meta_sampler(model.get_parameters(), z)
        energy, forces = model(x_pred, z)
        loss = energy.sum()
        loss.backward()
        optimizer.step()
        tqdm.write(f"loss: {loss.item()}")
