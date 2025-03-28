import copy

import numpy as np
import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        # sinusoidal functions using frequencies spaced logarithmically
        # a 10000^(2i/embedding_dim) scale
        half_size = self.size // 2
        emb = torch.log(torch.Tensor([10000.0]).to(x.device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size).to(x.device))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size

class PeriodicAngleEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        """
        Embedding for periodic angles in the range [-π, π].
        
        Args:
            size: The output embedding size (must be even)
            scale: Optional scaling factor for the input angles
        """
        super().__init__()
        assert size % 2 == 0, "Embedding size must be even"
        self.size = size
        self.scale = scale
        
    def forward(self, x: torch.Tensor):
        """
        Embeds periodic angles using sin and cos components.
        
        Args:
            x: Tensor of angles in radians, expected to be in range [-π, π]
        
        Returns:
            Embedded representation of the angles
        """
        # Scale the input if needed
        x = x * self.scale
        
        # Convert angles to sin/cos representation to handle periodicity
        sin_x = torch.sin(x)
        cos_x = torch.cos(x)
        
        # For larger embeddings, add higher frequency components
        # higher frequency components still respect the [-π, π] periodicity.
        # When using sin(n*θ) and cos(n*θ) where n is a frequency multiplier,
        # these functions remain periodic with respect to θ.
        # The period becomes 2π/n, which is still also 2π periodic
        # any input angle θ and θ+2π will produce identical embeddings for any frequency multiplier
        if self.size > 2:
            half_size = self.size // 2
            # Create frequency multipliers (1, 2, 3, ..., half_size)
            freqs = torch.arange(1, half_size + 1).to(x.device).float()
            
            # Expand dimensions for broadcasting
            angles = x.unsqueeze(-1)  # [batch, 1]
            freqs = freqs.unsqueeze(0)  # [1, half_size]
            
            # Compute sin and cos at different frequencies
            angles_mul = angles * freqs  # [batch, half_size]
            sin_components = torch.sin(angles_mul)  # [batch, half_size]
            cos_components = torch.cos(angles_mul)  # [batch, half_size]
            
            # Interleave sin and cos components
            embedded = torch.stack((sin_components, cos_components), dim=-1)
            embedded = embedded.flatten(start_dim=-2)  # [batch, size]
            return embedded
        else:
            # Simple case: just return sin and cos of the angle
            return torch.cat((sin_x.unsqueeze(-1), cos_x.unsqueeze(-1)), dim=-1)
    
    def __len__(self):
        return self.size

class PeriodicLearnableEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        """
        Learnable embedding for periodic angles in the range [-π, π].
        
        Args:
            size: The output embedding size (must be even)
            scale: Optional scaling factor for the input angles
        """
        super().__init__()
        assert size % 2 == 0, "Embedding size must be even"
        self.size = size
        self.scale = scale
        
        # Learnable parameters for transforming the sin/cos components
        half_size = size // 2
        self.sin_transform = nn.Linear(half_size, half_size)
        self.cos_transform = nn.Linear(half_size, half_size)
        
    def forward(self, x: torch.Tensor):
        """
        Embeds periodic angles using learnable transformations of sin and cos components.
        
        Args:
            x: Tensor of angles in radians, expected to be in range [-π, π]
        
        Returns:
            Embedded representation of the angles
        """
        # Scale the input if needed
        x = x * self.scale
        
        if self.size > 2:
            half_size = self.size // 2
            # Create frequency multipliers (1, 2, 3, ..., half_size)
            freqs = torch.arange(1, half_size + 1).to(x.device).float()
            
            # Expand dimensions for broadcasting
            angles = x.unsqueeze(-1)  # [batch, 1]
            freqs = freqs.unsqueeze(0)  # [1, half_size]
            
            # Compute sin and cos at different frequencies
            angles_mul = angles * freqs  # [batch, half_size]
            sin_components = torch.sin(angles_mul)  # [batch, half_size]
            cos_components = torch.cos(angles_mul)  # [batch, half_size]
            
            # Apply learnable transformations
            sin_transformed = self.sin_transform(sin_components)
            cos_transformed = self.cos_transform(cos_components)
            
            # Interleave sin and cos components
            embedded = torch.stack((sin_transformed, cos_transformed), dim=-1)
            embedded = embedded.flatten(start_dim=-2)  # [batch, size]
            return embedded
        else:
            # Simple case: just return sin and cos of the angle with learnable transformation
            sin_x = torch.sin(x).unsqueeze(-1)  # [batch, 1]
            cos_x = torch.cos(x).unsqueeze(-1)  # [batch, 1]
            
            sin_transformed = self.sin_transform(sin_x)
            cos_transformed = self.cos_transform(cos_x)
            
            return torch.cat((sin_transformed, cos_transformed), dim=-1)
    
    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: torch.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        elif type == "periodic_angle":
            self.layer = PeriodicAngleEmbedding(size, **kwargs)
        elif type == "periodic_learnable":
            self.layer = PeriodicLearnableEmbedding(size, **kwargs)
        else:
            raise ValueError(f"Unknown positional embedding type: {type}.")

    def forward(self, x: torch.Tensor):
        return self.layer(x)


class Block(nn.Module):
    def __init__(
        self, size: int, t_emb_size: int = 0, add_t_emb=False, concat_t_emb=False
    ):
        super().__init__()

        in_size = size + t_emb_size if concat_t_emb else size
        self.ff = nn.Linear(in_size, size)
        self.act = nn.GELU()

        self.add_t_emb = add_t_emb
        self.concat_t_emb = concat_t_emb

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor):
        in_arg = torch.cat([x, t_emb], dim=-1) if self.concat_t_emb else x
        out = x + self.act(self.ff(in_arg))

        if self.add_t_emb:
            out = out + t_emb

        return out


class FourierMLP(nn.Module):
    def __init__(
        self,
        in_shape=2,
        out_shape=2,
        num_layers=2,
        channels=128,
        zero_init=True,
        energy_function=None,
    ):
        super().__init__()

        self.in_shape = (in_shape,)
        self.out_shape = (out_shape,)

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(in_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(self.out_shape))),
        )
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
            self.layers[-1].bias.data.fill_(0.0)

    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_cond)
        return out.view(-1, *self.out_shape)


class TimeConder(nn.Module):
    def __init__(self, channel, out_dim, num_layers):
        super().__init__()
        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channel)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channel)[None])
        self.layers = nn.Sequential(
            nn.Linear(2 * channel, channel),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(channel, channel),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(channel, out_dim),
        )

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.01)

    def forward(self, t):
        if t.ndim < self.timestep_coeff.ndim:
            t = t.unsqueeze(-1)
        sin_cond = torch.sin((self.timestep_coeff * t.float()) + self.timestep_phase)
        cos_cond = torch.cos((self.timestep_coeff * t.float()) + self.timestep_phase)
        cond = rearrange([sin_cond, cos_cond], "d b w -> b (d w)")
        return self.layers(cond)


class MyMLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) with time and position embeddings.
    
    This MLP is designed to process both spatial coordinates and time information,
    with options for different embedding strategies and network architectures.
    
    Args:
        hidden_size (int): Size of the hidden layers. Default: 128.
        hidden_layers (int): Number of hidden layers. Default: 3.
        emb_size (int): Size of the embedding vectors. Default: 128.
        out_dim (int): Dimension of the output. Default: 2.
        time_emb (str): Type of time embedding ('sinusoidal', etc.). Default: 'sinusoidal'.
        input_emb (str): Type of input embedding ('sinusoidal', etc.). Default: 'sinusoidal'.
        add_t_emb (bool): Whether to add time embeddings to hidden representations. Default: False.
        concat_t_emb (bool): Whether to concatenate time embeddings before final layer. Default: False.
        input_dim (int): Dimension of the input coordinates. Default: 2.
        energy_function: Optional energy function for conditioning. Default: None.
        periodic_output (str, optional): Method to bound output to [-π, π] range. Options:
            'atan2' - Apply atan2 to pairs of outputs
            'sin_cos' - Predict sin/cos pairs and convert to angles
            'tanh' - Apply tanh and scale by π
            None - Unbounded output (default)
    """
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        out_dim: int = 2,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
        add_t_emb: bool = False,
        concat_t_emb: bool = False,
        input_dim: int = 2,
        energy_function=None,
        periodic_output: str = None,
    ):
        super().__init__()
        
        # assert hidden_size == emb_size, \
        #     f"hidden_size {hidden_size} must be equal to emb_size {emb_size}"

        self.add_t_emb = add_t_emb
        self.concat_t_emb = concat_t_emb
        self.periodic_output = periodic_output
        self.out_dim = out_dim

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)

        positional_embeddings = []
        for i in range(input_dim):
            embedding = PositionalEmbedding(emb_size, input_emb, scale=25.0)

            self.add_module(f"input_mlp{i}", embedding)

            positional_embeddings.append(embedding)

        self.channels = 1
        self.self_condition = False
        concat_size = len(self.time_mlp.layer) + sum(
            map(lambda x: len(x.layer), positional_embeddings)
        )

        layers = [nn.Linear(concat_size, hidden_size)]
        # middle layers
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size, emb_size, add_t_emb, concat_t_emb))

        # final layer
        in_size = emb_size + hidden_size if concat_t_emb else emb_size
        layers.append(nn.Linear(in_size, out_dim))

        self.layers = layers
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, t, x, x_self_cond=False):
        """
        Forward pass through the MLP.
        
        Args:
            t (torch.Tensor): Time tensor of shape (batch_size, 1).
            x (torch.Tensor): Input coordinates of shape (batch_size, input_dim).
            x_self_cond (bool or torch.Tensor): Self-conditioning input (not used in current implementation).
                Default: False.
                
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_dim).
        """
        # Create embeddings for each input dimension using the corresponding positional embedding layer
        positional_embs = [
            self.get_submodule(f"input_mlp{i}")(x[:, i]) for i in range(x.shape[-1])
        ]

        # Create time embedding
        t_emb = self.time_mlp(t.squeeze())
        
        # Concatenate all positional embeddings with time embedding
        x = torch.cat((*positional_embs, t_emb), dim=-1)

        # Process through network layers
        for i, layer in enumerate(self.layers):
            if i == 0:
                # First layer: apply linear transformation followed by GELU activation
                x = nn.GELU()(layer(x))
                # Optionally add time embedding (residual connection)
                if self.add_t_emb:
                    x = x + t_emb

            elif i == len(self.layers) - 1:
                # Final layer: optionally concatenate time embedding before final projection
                if self.concat_t_emb:
                    x = torch.cat([x, t_emb], dim=-1)

                # Apply final linear layer to get output
                x = layer(x)

            else:
                # Middle layers: apply Block which handles time embedding internally
                x = layer(x, t_emb)

        # x is same shape as input x [B, D]
        
        # Apply periodic output transformation if specified
        if self.periodic_output == 'atan2':
            x = torch.atan2(x, torch.ones_like(x) * torch.pi) * 2
        
        # elif self.periodic_output == 'sin_cos':
        #     # Split output into sin and cos components
        #     sin_out, cos_out = x.chunk(2, dim=-1)
            
        #     # Normalize to ensure unit circle
        #     norm = torch.sqrt(sin_out**2 + cos_out**2 + 1e-8)
        #     sin_out = sin_out / norm
        #     cos_out = cos_out / norm
            
        #     # Convert to angles
        #     x = torch.atan2(sin_out, cos_out)
        
        elif self.periodic_output == 'tanh':
            # Scale tanh output to [-π, π] range
            x = torch.pi * torch.tanh(x)

        return x


class MyMLPNoSpaceEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        out_dim: int = 2,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
        add_t_emb: bool = False,
        concat_t_emb: bool = False,
    ):
        super().__init__()

        self.add_t_emb = add_t_emb
        self.concat_t_emb = concat_t_emb

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.channels = 1
        self.self_condition = False
        concat_size = len(self.time_mlp.layer) + 2
        layers = [nn.Linear(concat_size, hidden_size)]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size, emb_size, add_t_emb, concat_t_emb))

        in_size = emb_size + hidden_size if concat_t_emb else emb_size
        layers.append(nn.Linear(in_size, out_dim))

        self.layers = layers
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, t, x, x_self_cond=False):
        t_emb = self.time_mlp(t.squeeze())
        x = torch.cat((x, t_emb), dim=-1)

        for i, layer in enumerate(self.layers):
            if i == 0:
                x = nn.GELU()(layer(x))
                if self.add_t_emb:
                    x = x + t_emb

            elif i == len(self.layers) - 1:
                if self.concat_t_emb:
                    x = torch.cat([x, t_emb], dim=-1)

                x = layer(x)

            else:
                x = layer(x, t_emb)

        return x


class MyMLPNoEmbedding(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        out_dim: int = 2,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
        add_t_emb: bool = False,
        concat_t_emb: bool = False,
    ):
        super().__init__()

        self.add_t_emb = add_t_emb
        self.concat_t_emb = concat_t_emb

        emb_size = 1
        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.channels = 1
        self.self_condition = False
        concat_size = 3
        layers = [nn.Linear(concat_size, hidden_size)]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size, emb_size, add_t_emb, concat_t_emb))

        in_size = emb_size + hidden_size if concat_t_emb else emb_size
        layers.append(nn.Linear(in_size, out_dim))

        self.layers = layers
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, t, x, x_self_cond=False):
        # t_emb = self.time_mlp(t.squeeze())
        t_emb = t.unsqueeze(1)
        x = torch.cat((x, t_emb), dim=-1)

        for i, layer in enumerate(self.layers):
            if i == 0:
                x = nn.GELU()(layer(x))
                if self.add_t_emb:
                    x = x + t_emb

            elif i == len(self.layers) - 1:
                if self.concat_t_emb:
                    x = torch.cat([x, t_emb], dim=-1)

                x = layer(x)

            else:
                x = layer(x, t_emb)

        return x


class MyMLP6dim(nn.Module):
    def __init__(
        self,
        hidden_size: int = 128,
        hidden_layers: int = 3,
        emb_size: int = 128,
        time_emb: str = "sinusoidal",
        input_emb: str = "sinusoidal",
    ):
        super().__init__()

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp3 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp4 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp5 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp6 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        self.channels = 1
        self.self_condition = False
        concat_size = (
            len(self.time_mlp.layer)
            + len(self.input_mlp1.layer)
            + len(self.input_mlp2.layer)
            + len(self.input_mlp3.layer)
            + len(self.input_mlp4.layer)
            + len(self.input_mlp5.layer)
            + len(self.input_mlp6.layer)
        )
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        for _ in range(hidden_layers):
            layers.append(Block(hidden_size))
        layers.append(nn.Linear(hidden_size, 6))
        self.joint_mlp = nn.Sequential(*layers)

    def forward(self, t, x, x_self_cond=False):
        x1_emb = self.input_mlp1(x[:, 0])
        x2_emb = self.input_mlp2(x[:, 1])
        x3_emb = self.input_mlp3(x[:, 2])
        x4_emb = self.input_mlp4(x[:, 3])
        x5_emb = self.input_mlp5(x[:, 4])
        x6_emb = self.input_mlp6(x[:, 5])
        t_emb = self.time_mlp(t.squeeze())
        x = torch.cat((x1_emb, x2_emb, x3_emb, x4_emb, x5_emb, x6_emb, t_emb), dim=-1)
        x = self.joint_mlp(x)
        return x


class SpectralNormMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 2,
        hidden1_size: int = 64,
        hidden2_size: int = 128,
        output_size: int = 1,
    ):
        super().__init__()

        # First hidden layer with spectral normalization
        self.fc1 = spectral_norm(nn.Linear(input_size, hidden1_size))

        # Second hidden layer with spectral normalization
        self.fc2 = spectral_norm(nn.Linear(hidden1_size, hidden2_size))

        # Output layer with spectral normalization
        self.fc3 = spectral_norm(nn.Linear(hidden2_size, output_size))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
