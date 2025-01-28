import torch
from torch import nn
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# toy ML force field

def target_f(x):
    """
    """
    return torch.sin(x) + torch.sin((10./3.)*x)
    # return torch.cos(x) + x**8 - 0.5*x**2




class ToyMLFF(torch.nn.Module):
    def __init__(self, n_hidden=64):
        super().__init__()
        # Input features: flattened positions (3*n_atoms)
        s = n_hidden
        self.net = nn.Sequential(
            nn.Linear(1, s),
            nn.ReLU(),
            nn.Linear(s, s),
            nn.ReLU(),
            nn.Linear(s, s),
            nn.ReLU(),            
            nn.Linear(s, 1)
        )

    def forward(self, x):
        """
        x
        return: f(x)
        """
        return self.net(x)
    
    def get_parameters(self):
        """Returns flattened parameters"""
        return torch.cat([p.flatten() for p in self.parameters()])

    
class MetaSampler(torch.nn.Module):
    """Predicts the atom positions x that minimize the energy E(x).
    Takes as input the model parameters and atom numbers z.
    Returns the predicted positions.
    """
    def __init__(self, n_params):
        super().__init__()
        self.n_params = n_params
        
        # simple MLP to predict the positions
        self.net = nn.Sequential(
            nn.Linear(n_params, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),            
            nn.Linear(128, 1),
        )
    
    def forward(self, params):
        """
        params: model parameters (n_params)
        return: predicted f_min
        """
        params = params.reshape(1, -1)
        x = self.net(params)
        return x

if __name__ == "__main__":
    # seed
    # torch.manual_seed(1)
    
    TRAIN=True# False

    if TRAIN:
        xs = torch.linspace(-1, 1, 500).reshape(-1, 1)
        ys = target_f(xs)
        sns.lineplot(x=xs.squeeze(), y=ys.squeeze())
        model = ToyMLFF()
        # train model to predic target_f using (xs, ys) as training data
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        dataset = torch.utils.data.TensorDataset(xs, ys)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
        for _ in tqdm(range(10000)):
            for x, y in dataloader:
                optimizer.zero_grad()
                y_pred = model(x)
                loss = torch.nn.functional.mse_loss(y_pred, y)
                loss.backward()
                optimizer.step()


        # save parameters:
        torch.save(model.state_dict(), "model_doublewell.pth")

        xs = torch.linspace(-1.1, 1.1, 500).reshape(-1, 1)
        pred_ys = model(xs).detach().squeeze()
        sns.lineplot(x=xs.squeeze(), y=pred_ys)
        plt.show()
        exit()

    else:
        model = ToyMLFF()
        model.load_state_dict(torch.load("model.pth"))

        xs = torch.linspace(-10, 15, 100).reshape(-1, 1)
        ys = target_f(xs)
        sns.lineplot(x=xs.squeeze(), y=ys.squeeze())
        pred_ys = model(xs).detach().squeeze()
        sns.lineplot(x=xs.squeeze(), y=pred_ys)
        # plt.show()


    n_params = model.get_parameters().shape[0]
    meta_sampler = MetaSampler(n_params)
    x_pred = meta_sampler(model.get_parameters()).detach().reshape(-1)
    sns.scatterplot(x=x_pred, y=target_f(x_pred), color="blue")
    
    # train meta sampler
    optimizer = torch.optim.Adam(meta_sampler.parameters(), lr=1e-3)
    for _ in tqdm(range(100)):
        optimizer.zero_grad()
        x_pred = meta_sampler(model.get_parameters())
        x = model(x_pred)
        loss = x.sum()
        loss.backward()
        optimizer.step()
        tqdm.write(f"loss: {loss.item()}")
        

    x_pred = meta_sampler(model.get_parameters()).detach().reshape(-1)
    sns.scatterplot(x=x_pred, y=target_f(x_pred), color="red")

    # model_1 = ToyMLFF()
    # model_1.load_state_dict(torch.load("model_1.pth"))
    # x_pred = meta_sampler(model_1.get_parameters()).detach().reshape(-1)
    # sns.scatterplot(x=x_pred, y=target_f(x_pred), color="red", marker="x")
    
    plt.show()
    

    
