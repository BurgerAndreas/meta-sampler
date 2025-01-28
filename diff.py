from tqdm import tqdm
import torch
from torch import nn

import seaborn as sns
import matplotlib.pyplot as plt


class MetaSampler(torch.nn.Module):
    """Predicts the atom positions x that minimize the energy E(x).
    Takes as input the model parameters and atom numbers z.
    Returns the predicted positions.
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        
        # simple MLP to predict the positions
        self.net = nn.Sequential(
            nn.Linear(n_in, n_in),
            nn.ReLU(),
            # nn.Linear(n_in, n_in),
            # nn.ReLU(),            
            nn.Linear(n_in, n_out),
            nn.ReLU(),            
            nn.Linear(n_out, n_out),
        )
    
    def forward(self, params):
        """
        params: model parameters (n_params)
        return: predicted f_min
        """
        params = params.reshape(1, -1)
        x = self.net(params)
        return x


class Diffusion(nn.Module):
    """
    diffusion model that outputs a number

    
    """

    def __init__(self, beta_min=1e-3, beta_max=1.0):
        super(Diffusion, self).__init__()
        self.score_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.register_parameter('beta_min', nn.Parameter(torch.tensor(beta_min)))
        self.register_parameter('beta_max', nn.Parameter(torch.tensor(beta_max)))
        

    def prior(self):
        return torch.normal(0, 1, (64,1))

    def noise(self):
        return torch.normal(0, 1, (64,1))

    def beta(self, t):
        return self.beta_min + (self.beta_max - self.beta_min) * t
        

    def score(self, x, t):
        t = torch.ones_like(x) * t
        x = torch.cat([x, t], dim=-1)
        return self.score_net(x)
    
    def sample(self, steps=25):
        x = self.prior()
        ts = torch.linspace(1, 1e-3, steps)
        dt = ts[0] - ts[1]
        for t in ts:
            x = x + self.beta(t)**2 / 2 * self.score(x, t) * dt + self.beta(t) * torch.sqrt(dt) * self.noise()

        return x

    def get_parameters(self):
        """Returns flattened parameters"""
        return torch.cat([p.flatten() for p in self.parameters()])

    def set_parameters(self, params):
        """Sets parameters from flattened params"""
        params = params.flatten()
        i = 0
        for p in self.parameters():
            n = p.numel()
            p.data = params[i:i+n].reshape(p.shape)
            i += n


if __name__ == "__main__":
    # seed
    torch.manual_seed(43)

    from toy_example import ToyMLFF, target_f
    
    model = ToyMLFF()
    model.load_state_dict(torch.load("model.pth"))

    # xs = torch.linspace(-1.1, 1.1, 500).reshape(-1, 1)
    xs = torch.linspace(-10, 15, 100).reshape(-1, 1)
    ys = target_f(xs)

    fig, ax = plt.subplots(nrows=2, sharex=True, height_ratios=[1, 3])
    sns.lineplot(x=xs.squeeze(), y=ys.squeeze(), ax=ax[1])
    pred_ys = model(xs).detach().squeeze()
    sns.lineplot(x=xs.squeeze(), y=pred_ys, ax=ax[1])


    diff_model = Diffusion()
    

    n_in = model.get_parameters().shape[0]
    n_out = diff_model.get_parameters().shape[0]
    meta_sampler = MetaSampler(n_in, n_out)

    # x = diff_model.sample()
    # print(x)
    
    # x_pred = meta_sampler(model.get_parameters()).detach().reshape(-1)
    # sns.scatterplot(x=x_pred, y=target_f(x_pred), color="blue")
    
    # train meta sampler
    # optimizer = torch.optim.Adam(meta_sampler.parameters(), lr=1e-3)
    optimizer = torch.optim.Adam(diff_model.parameters(), lr=1e-3)
    for _ in tqdm(range(1000)):
        optimizer.zero_grad()
        
        # weights = meta_sampler(model.get_parameters())
        # diff_model.set_parameters(weights)
        
        x_pred = diff_model.sample()
        y = model(x_pred)
        
        loss = y.sum()
        
        loss.backward()
        optimizer.step()
        tqdm.write(f"loss: {loss.item()}")
        

    # weights = meta_sampler(model.get_parameters())
    # diff_model.set_parameters(weights)

    xs_pred = []
    for _ in range(200):
        x_pred = diff_model.sample(steps=25)
        xs_pred.append(x_pred)
        sns.scatterplot(x=x_pred.detach().reshape(-1), y=target_f(x_pred).detach().reshape(-1), color="red", ax=ax[1])

    sns.histplot(torch.cat(xs_pred).detach().reshape(-1), color="green", alpha=0.3, bins=xs.reshape(-1), ax=ax[0])


    # gradient of boltzman dist with respect to x
    xs.requires_grad = True
    boltzman_dist = torch.exp(-model(xs))
    grad = torch.autograd.grad(boltzman_dist.sum(), xs)[0]
    xs = xs.detach()

    sns.lineplot(x=xs.squeeze(), y=boltzman_dist.detach().squeeze(), color="green", dashes=True, ax=ax[1])
    # sns.lineplot(x=xs.squeeze(), y=grad.detach().squeeze(), color="blue", dashes=True, ax=ax[1])

    
    # model_1 = ToyMLFF()
    # model_1.load_state_dict(torch.load("model_1.pth"))
    # x_pred = meta_sampler(model_1.get_parameters()).detach().reshape(-1)
    # sns.scatterplot(x=x_pred, y=target_f(x_pred), color="red", marker="x")

    # score = diff_model.score(xs, 0.5).detach().reshape(-1)
    # sns.lineplot(x=xs.squeeze(), y=score.squeeze(), color="red", ax=ax[1])
    
    print(diff_model.beta_min, diff_model.beta_max)
    plt.show()

    

    
    
