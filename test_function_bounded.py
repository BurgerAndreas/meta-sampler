import torch
import torch.nn as nn
import plotly.graph_objects as go

"""
Key points about global minimum existence:

1. When coefficients[3] (quartic term) is positive:
   - The function always has a global minimum because the quartic term dominates
     as |x| → ∞
   - The sinusoidal term creates local minima but can't overcome the quartic growth

2. When coefficients[3] is zero or negative:
   - If zero: The function might not have a global minimum if coefficients[2] ≤ 0
   - If negative: The function goes to -∞ as |x| → ∞, so no global minimum exists

3. The sinusoidal terms (coefficients[0], coefficients[1]):
   - Control the number and depth of local minima
   - Don't affect the existence of a global minimum
   - Larger values create more pronounced local minima

4. The bias term (coefficients[4]):
   - Shifts the function vertically
   - Doesn't affect the existence or location of minima
"""

class MultiMinima(nn.Module):
    def __init__(self, coefficients=None, random_seed=None, n_sines=2):
        super().__init__()
        if random_seed is not None:
            torch.manual_seed(random_seed)
            
        self.n_sines = n_sines if coefficients is None else (len(coefficients)-2)//3
        
        # Default coefficients if none provided
        # Format: [amplitude_1, freq_1, offset_1, ..., amplitude_n, freq_n, offset_n, quadratic, bias]
        if coefficients is None:
            coefficients = MultiMinima.sample_coefficients(n_samples=1, n_sines=n_sines)[0]
            
        self.coefficients = torch.tensor(coefficients, requires_grad=True)
    
    @staticmethod
    def sample_coefficients(n_samples=1, n_sines=2):
        """
        Sample coefficients for the function.
        
        Args:
            n_samples (int): Number of coefficient sets to sample
            n_sines (int): Number of sinusoidal components
        
        Returns:
            torch.Tensor: Sampled coefficients of shape (n_samples, 3*n_sines + 2)
        """
        # Sample coefficients for sines only (amplitude, frequency, offset for each)
        coeffs = torch.randn(n_samples, 3*n_sines)
        
        # Scale sine frequencies, amplitudes and offsets
        for i in range(n_sines):
            # Increase frequencies for more local minima
            coeffs[:, 3*i + 1] = coeffs[:, 3*i + 1] * (i + 1.0) * 2.0
            # Scale amplitudes
            coeffs[:, 3*i] = coeffs[:, 3*i] * 3.0 / (i + 1.0)
            # Keep random offsets in reasonable range
            coeffs[:, 3*i + 2] = coeffs[:, 3*i + 2] * 2.0
            
        # Add fixed quadratic and bias terms
        quadratic_term = torch.full((n_samples, 1), 3.0)
        bias_term = torch.full((n_samples, 1), -1.0)
        coeffs = torch.cat([coeffs, quadratic_term, bias_term], dim=1)
            
        return coeffs

    def forward(self, x):
        """
        Creates a function with multiple local minima using multiple sinusoidal terms
        and a quadratic term to ensure global minimum.
        """
        # Sum multiple sinusoidal components
        sin_term = 0
        for i in range(self.n_sines):
            amplitude = self.coefficients[3*i]
            frequency = self.coefficients[3*i + 1]
            offset = self.coefficients[3*i + 2]
            sin_term += amplitude * torch.sin(frequency * (x + offset))
        
        # # Option 1: Quadratic term to ensure global minimum
        # quad_term = self.coefficients[-2] * x**2 + self.coefficients[-1]
        
        flat_region = 4.0  # Width of flat region around x=0
        
        # Option 2: Modified quadratic term that's flat around zero
        x_abs = torch.abs(x)
        quad_mask = x_abs > flat_region
        quad_term = torch.zeros_like(x)
        quad_term[quad_mask] = self.coefficients[-2] * ((x_abs[quad_mask] - flat_region) ** 2)
        quad_term = quad_term + self.coefficients[-1]
        
        # # Option 3: Quadratic term with smooth transition around zero
        # x_abs = torch.abs(x)
        # # Smooth transition using sigmoid
        # transition = torch.sigmoid((x_abs - flat_region) * 5) # 5 controls transition sharpness
        # quad_base = (x_abs - flat_region) ** 2
        # quad_term = self.coefficients[-2] * quad_base * transition + self.coefficients[-1]
        
        return sin_term + quad_term

def plot_function(model, title, filename, predicted_min=None, plot_individual_terms=False):
    """Plot the function with its global minimum"""
    # Create input range for visualization
    x = torch.linspace(-5, 5, 200)
    
    # Compute function values
    with torch.no_grad():
        y = model(x)
    
    # Find global minimum
    global_min_idx = torch.argmin(y)
    print(f"\nGlobal minimum:")
    print(f"x = {x[global_min_idx]:.3f}, f(x) = {y[global_min_idx]:.3f}")

    # Create plotly figure
    fig = go.Figure()

    # Add main function curve
    fig.add_trace(go.Scatter(
        x=x.numpy(),
        y=y.numpy(),
        name='Total Function',
        mode='lines'
    ))

    if plot_individual_terms:
        # Plot individual sine terms
        for i in range(model.n_sines):
            amplitude = model.coefficients[3*i]
            frequency = model.coefficients[3*i + 1]
            offset = model.coefficients[3*i + 2]
            sine_term = amplitude * torch.sin(frequency * (x + offset))
            fig.add_trace(go.Scatter(
                x=x.numpy(),
                y=sine_term.detach().cpu().numpy(),
                name=f'Sine Term {i+1}',
                mode='lines',
                line=dict(dash='dash')
            ))

        # Plot quadratic term
        flat_region = 4.0
        x_abs = torch.abs(x)
        quad_mask = x_abs > flat_region
        quad_term = torch.zeros_like(x)
        quad_term[quad_mask] = model.coefficients[-2] * ((x_abs[quad_mask] - flat_region) ** 2)
        quad_term = quad_term + model.coefficients[-1]
        
        fig.add_trace(go.Scatter(
            x=x.numpy(),
            y=quad_term.detach().cpu().numpy(),
            name='Quadratic Term',
            mode='lines',
            line=dict(dash='dot')
        ))

    # Add global minimum point
    fig.add_trace(go.Scatter(
        x=[x[global_min_idx].item()],
        y=[y[global_min_idx].item()],
        mode='markers',
        name='Global Minimum',
        marker=dict(color='green', size=12)
    ))
    
    if predicted_min is not None:
        fig.add_trace(go.Scatter(
            x=[predicted_min],
            y=[model(torch.tensor(predicted_min))],
            mode='markers',
            name='Predicted Minimum',
            marker=dict(color='red', size=12)
        ))
    

    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title='x',
        yaxis_title='f(x)',
        template='plotly_white',
        showlegend=True
    )

    # Save the plot
    fig.write_image(filename)
    print(f"Saved plot to {filename}")

# Example usage
if __name__ == "__main__":
    # Example 1: Random sampling with guaranteed global minimum
    sampled_coeffs = MultiMinima.sample_coefficients(n_samples=10)
    print("\nSampled coefficients (with guaranteed global minimum):")
    print(sampled_coeffs)
    
    # Create models with different sampled coefficients
    for i, coeffs in enumerate(sampled_coeffs):
        model = MultiMinima(coeffs.tolist())
        plot_function(
            model,
            title=f'Multi-Minima Function (Model {i+1})',
            filename=f"test_function/multi_minima_function_{i+1}.png",
            plot_individual_terms=True
        )

