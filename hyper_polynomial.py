import torch
import torch.nn as nn
from tqdm import tqdm
from test_function_bounded import MultiMinima, plot_function
import numpy as np

# Neural Network Model
class PolynomialMinFinder(nn.Module):
    def __init__(self, input_size=5):  # Changed to match MultiMinima coefficients size
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# Training function
def train_min_finder(
    model, num_epochs=1000, batch_size=64,
    coeffs_overfit=None, device=None, dtype=torch.float32,
    lr=1e-4, dataset_size=-1
    ):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if dataset_size > 0:
        dataset = MultiMinima.sample_coefficients(n_samples=dataset_size)
        dataset = dataset.to(device=device, dtype=dtype)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in tqdm(range(num_epochs)):
        if coeffs_overfit is not None:
            # all epochs/iterations will use the same batch
            coeffs = coeffs_overfit
        elif dataset_size > 0:
            # index into the dataset
            coeffs = dataset[epoch % len(dataset)]
        else:
            # Generate random coefficients for this batch
            coeffs = MultiMinima.sample_coefficients(n_samples=batch_size)
            coeffs = coeffs.to(device=device, dtype=dtype)
        
        # Forward pass
        predicted_mins = model(coeffs)
        
        # Create test function instances and compute loss
        test_functions = [MultiMinima(c.tolist()) for c in coeffs]
        loss = torch.mean(torch.stack([f(min_x) for f, min_x in zip(test_functions, predicted_mins)]))
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            tqdm.write(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Main execution
def main():
    # Set parameters
    DTYPE = torch.float32
    BATCH_SIZE = 64
    NUM_EPOCHS = 10000
    overfit_to_single_batch = False
    dataset_size = -1 # -1 means a new random batch every iteration, no repeats
    n_sines = 2
    n_coeffs = n_sines*3 + 2
    seed = 42
    lr = 1e-4
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = PolynomialMinFinder(input_size=n_coeffs).to(dtype=DTYPE)
    model = model.to(device)
    
    if overfit_to_single_batch:
        coeffs = MultiMinima.sample_coefficients(n_samples=BATCH_SIZE, n_sines=n_sines).to(device=device, dtype=DTYPE)
        print("coeffs", list(coeffs.shape))
        
        # Plot initial function
        test_function = MultiMinima(coeffs[0].tolist())
        plot_function(test_function, "Initial Function", "initial_function.png")
    else:
        coeffs = None
    
    # Train the model
    train_min_finder(
        model, num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        coeffs_overfit=coeffs,
        device=device, dtype=DTYPE,
        lr=lr,
        dataset_size=dataset_size
    )
    
    # Test the model
    model.eval()
    with torch.no_grad():
        if overfit_to_single_batch:
            # test on the same coefficients as the training set
            test_coeffs = coeffs[0].detach()
        else:
            # Generate test coefficients
            test_coeffs = MultiMinima.sample_coefficients(n_samples=1)[0].to(device=device, dtype=DTYPE)
        
        # Get prediction
        predicted_min = model(test_coeffs).cpu().item()
        
        # Create test function and plot result
        test_function = MultiMinima(test_coeffs.cpu().tolist())
        plot_function(
            test_function, "Test Function with Predicted Minimum", "test_function_result.png",
            predicted_min=predicted_min
        )
        print(f"\nPredicted minimum at x = {predicted_min:.4f}")

if __name__ == "__main__":
    main()