import torch
import torch.nn as nn
import torch.nn.functional as F

class AngleMLP(nn.Module):
    def __init__(self, hidden_size=64):
        super(AngleMLP, self).__init__()
        # The input is two angles, each represented as (sin, cos) → 4 features total.
        self.fc1 = nn.Linear(4, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # The output layer predicts 4 values corresponding to two (sin, cos) pairs.
        self.fc3 = nn.Linear(hidden_size, 4)
    
    def forward(self, angles):
        """
        angles: Tensor of shape (batch_size, 2) containing angles in radians.
        Returns:
            angles_out: Tensor of shape (batch_size, 2) containing the predicted angles.
        """
        # Embed the input angles as sin and cos components.
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)
        x = torch.cat([sin_angles, cos_angles], dim=-1)  # shape: (batch, 4)
        
        # Pass through the MLP.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # shape: (batch, 4)
        
        # Split the 4-dimensional output into two pairs.
        sin_out, cos_out = x.chunk(2, dim=-1)  # Each is (batch, 2)
        
        # Normalize each (sin, cos) pair to ensure they lie on the unit circle.
        norm = torch.sqrt(sin_out**2 + cos_out**2 + 1e-8)
        sin_out = sin_out / norm
        cos_out = cos_out / norm
        
        # Convert the normalized (sin, cos) pairs back to angles.
        angles_out = torch.atan2(sin_out, cos_out)
        return angles_out

# Example usage:
if __name__ == "__main__":
    with torch.no_grad():
        ###################################################################
        # AngleMLP (quick example here, not used in DEM)
        ###################################################################
        model = AngleMLP(hidden_size=64)
        
        # Example batch: 5 samples with 2 angles each (in radians).
        input_angles = torch.tensor([0.0, 1.57])
        # repeat and add random noise * 2pi
        input_angles = input_angles.repeat(5, 1) 
        periodic_noise = torch.randint(0, 10, (5, 1)) * 2 * torch.pi
        input_angles += periodic_noise
        
        # Forward pass.
        output_angles = model(input_angles)
        print("Input Angles:\n", input_angles)
        print("Output Angles (should be identical):\n", output_angles)
        
        
        ###################################################################
        # Default MLP in DEM
        ###################################################################
        print("-" * 100)
        print("Default MLP in DEM")
        
        from dem.models.components.mlp import MyMLP
        
        mlp = MyMLP(
            hidden_size=128,
            hidden_layers=3,
            emb_size=128,
            input_dim=2,
            out_dim=2,
            time_emb="sinusoidal",
            input_emb="sinusoidal",
            # add_t_emb=False,
            # concat_t_emb=False,
            # energy_function=None,
            # periodic_output=None,
        )

        t = torch.ones(input_angles.shape[0]).unsqueeze(1) * 0.5
        print("t.shape", t.shape)
        output_angles = mlp(t, input_angles)
        # output_angles = torch.vmap(mlp, in_dims=(0, 0))(t, input_angles)
        print("Output Angles (won't be identical):\n", output_angles)
        
        ###################################################################
        # MLP with new periodic angle input (but unbounded output)
        ###################################################################
        print("-" * 100)
        print("MLP with new periodic angle input (but unbounded output)")
        
        for input_emb in ["periodic_angle", "periodic_learnable"]:
            print(f"\ninput_emb: {input_emb}")
            mlp = MyMLP(
                hidden_size=128,
                hidden_layers=3,
                emb_size=128,
                input_dim=2,
                out_dim=2,
                time_emb="sinusoidal",
                input_emb=input_emb, # periodic_angle, periodic_learnable
                # add_t_emb=False,
                # concat_t_emb=False,
                # energy_function=None,
                # periodic_output=None, # atan2 tanh 
            )

            t = torch.ones(input_angles.shape[0]).unsqueeze(1) * 0.5
            print("t.shape", t.shape)
            output_angles = mlp(t, input_angles)
            # output_angles = torch.vmap(mlp, in_dims=(0, 0))(t, input_angles)
            print("Output Angles (should be identical):\n", output_angles)
            
            # try some random angles and see if they are in the range [-pi, pi]
            random_angles = torch.randn(10, 2) * 10.0
            t = torch.ones(random_angles.shape[0]).unsqueeze(1) * 0.5
            # set random MLP weights
            for param in mlp.parameters():
                param.data.uniform_(-10.0, 10.0)
            output_angles = mlp(t, random_angles)
            print("Output Angles range:\n", output_angles.min(), output_angles.max())
        
        ###################################################################
        # MLP with new periodic angle input and output bounded in [-pi, pi]
        ###################################################################
        print("-" * 100)
        print("MLP with new periodic angle input and output bounded in [-pi, pi]")
        
        for periodic_output in ["atan2", "tanh"]:
            print(f"\nperiodic_output: {periodic_output}")
            mlp = MyMLP(
                hidden_size=128,
                hidden_layers=3,
                emb_size=128,
                input_dim=2,
                out_dim=2,
                time_emb="sinusoidal",
                input_emb="periodic_angle", # periodic_angle, periodic_learnable
                # add_t_emb=False,
                # concat_t_emb=False,
                # energy_function=None,
                periodic_output=periodic_output, # atan2 tanh 
            )
            
            # set random MLP weights
            for param in mlp.parameters():
                param.data.uniform_(-10.0, 10.0)

            t = torch.ones(input_angles.shape[0]).unsqueeze(1) * 0.5
            output_angles = mlp(t, input_angles)
            # output_angles = torch.vmap(mlp, in_dims=(0, 0))(t, input_angles)
            print("Output Angles (should be identical):\n", output_angles)
        
            # try some random angles and see if they are in the range [-pi, pi]
            random_angles = torch.randn(10, 2) * 10.0
            t = torch.ones(random_angles.shape[0]).unsqueeze(1) * 0.5
            output_angles = mlp(t, random_angles)
            print("Output Angles range:\n", output_angles.min(), output_angles.max())