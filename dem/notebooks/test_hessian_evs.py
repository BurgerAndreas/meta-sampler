import torch

evs = [
    [torch.tensor([[-1.0, 0.5]]), "low"],
    [torch.tensor([[-1.0, -0.5]]), "high"],
    [torch.tensor([[0.5, 1.0]]), "high"],
]

hessian_eigenvalue_penalty = "softmax_mult"

for ev in evs:
    smallest_eigenvalues, ground_truth = ev
    if hessian_eigenvalue_penalty == "softplus":
        # [0, inf]
        # Using softplus which is differentiable everywhere but still creates one-sided penalties
        # if first eigenvalue > 0, increase energy
        ev1_bias = torch.nn.functional.softplus(smallest_eigenvalues[:, 0])
        # if second eigenvalue > 0, increase energy
        ev2_bias = torch.nn.functional.softplus(-smallest_eigenvalues[:, 1])
        saddle_bias = ev1_bias + ev2_bias
    elif hessian_eigenvalue_penalty == "relu":
        ev1_bias = torch.relu(smallest_eigenvalues[:, 0])
        ev2_bias = torch.relu(-smallest_eigenvalues[:, 1])
        saddle_bias = ev1_bias + ev2_bias
    # elif hessian_eigenvalue_penalty == 'heaviside':
    #     # 1 if smallest_eigenvalues[0] > 0 else 0
    #     ev1_bias = torch.heaviside(smallest_eigenvalues[:, 0], torch.tensor(0.))
    #     # 1 if smallest_eigenvalues[1] < 0 else 0
    #     ev2_bias = torch.heaviside(-smallest_eigenvalues[:, 1], torch.tensor(0.))
    #     saddle_bias = ev1_bias + ev2_bias
    elif hessian_eigenvalue_penalty == "sigmoid":
        ev1_bias = torch.sigmoid(smallest_eigenvalues[:, 0])
        ev2_bias = torch.sigmoid(-smallest_eigenvalues[:, 1])
        saddle_bias = ev1_bias + ev2_bias
    elif hessian_eigenvalue_penalty == "mult":
        # Penalize if both eigenvalues are positive or negative
        ev1_bias = smallest_eigenvalues[:, 0]
        ev2_bias = smallest_eigenvalues[:, 1]
        saddle_bias = ev1_bias * ev2_bias
    elif hessian_eigenvalue_penalty == "tanh_mult":
        # Penalize if both eigenvalues are positive or negative
        # [-1, 1]
        # tanh: ~ -1 for negative, 1 for positive
        # both neg -> 1, both pos -> 1, one neg one pos -> 0
        ev1_bias = torch.tanh(smallest_eigenvalues[:, 0])
        ev2_bias = torch.tanh(smallest_eigenvalues[:, 1])
        saddle_bias = ev1_bias * ev2_bias
        saddle_bias += 1.0  # [0, 2]
    print(f"Ground truth: {ground_truth}, Saddle bias: {saddle_bias}")
