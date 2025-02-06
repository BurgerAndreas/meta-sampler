import torch
import torch.nn as nn
import plotly.graph_objects as go

class MultiMinima(nn.Module):
    def __init__(self, coefficients=None):
        super().__init__()
        # Default coefficients if none provided
        if coefficients is None:
            coefficients = [1.0, -2.0, 0.5, 3.0, -1.0]
        self.coefficients = torch.tensor(coefficients, requires_grad=True)
    
    def forward(self, x):
        """
        Creates a function with multiple local minima using sinusoidal terms
        and polynomial terms.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Function value at x
        """
        # Sinusoidal components for multiple minima
        sin_term = self.coefficients[0] * torch.sin(self.coefficients[1] * x)
        
        # Polynomial term for overall shape
        poly_term = self.coefficients[2] * x**2 + \
                   self.coefficients[3] * x**4 + \
                   self.coefficients[4]
        
        return sin_term + poly_term

def plot_function(model, filename):
    """Plot the function with its minima"""
    # Create input range for visualization
    x = torch.linspace(-4, 4, 200)
    
    # Compute function values
    with torch.no_grad():
        y = model(x)
    
    # Find approximate minima
    potential_minima = []
    for i in range(1, len(y)-1):
        if y[i-1] > y[i] and y[i] < y[i+1]:
            potential_minima.append((x[i].item(), y[i].item()))
    
    print("Potential local minima (x, f(x)):")
    for x_min, y_min in potential_minima:
        print(f"x = {x_min:.3f}, f(x) = {y_min:.3f}")
    
    # Global minimum
    global_min_idx = torch.argmin(y)
    print(f"\nGlobal minimum:")
    print(f"x = {x[global_min_idx]:.3f}, f(x) = {y[global_min_idx]:.3f}")

    # Create plotly figure
    fig = go.Figure()

    # Add main function curve
    fig.add_trace(go.Scatter(
        x=x.numpy(),
        y=y.numpy(),
        name='Function',
        mode='lines'
    ))

    # Add local minima points
    if potential_minima:
        x_mins, y_mins = zip(*potential_minima)
        fig.add_trace(go.Scatter(
            x=list(x_mins),
            y=list(y_mins),
            mode='markers',
            name='Local Minima',
            marker=dict(color='red', size=10)
        ))

    # Add global minimum point
    fig.add_trace(go.Scatter(
        x=[x[global_min_idx].item()],
        y=[y[global_min_idx].item()],
        mode='markers',
        name='Global Minimum',
        marker=dict(color='green', size=12)
    ))

    # Update layout
    fig.update_layout(
        title='Multi-Minima Function',
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
    # Generate and plot multiple random functions
    n_samples = 3
    for i in range(n_samples):
        # Sample random coefficients
        coeffs = torch.randn(5)
        # Ensure positive quartic term for global minimum
        coeffs[3] = abs(coeffs[3]) + 0.1
        
        # Create and plot model
        model = MultiMinima(coeffs.tolist())
        plot_function(model, f"multi_minima_random_{i+1}.png")