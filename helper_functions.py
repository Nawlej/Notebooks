import torch
import numpy as np
import matplotlib.pyplot as plt

def shuffle(input: torch.Tensor, seed: int =None):
    if seed != None: 
        torch.manual_seed(seed)
    idx = torch.randperm(input.shape[0])
    
    return input.view(input.shape)[idx]


def plot_decision_boundaries(model: torch.nn.Module, data: torch.Tensor, labels: torch.Tensor):
    """ Plots the decision boundary in a 2D space for classification
    """
    data = data.detach().numpy()
    x = data[:, 0]
    y = data[:, 1]

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict the output for each point in the mesh grid
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        Z = model(grid_points).squeeze()
        if y.shape[-1] > 1: 
            Z = Z.argmax(1)
        Z = Z.reshape(xx.shape)
        
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.7) 
    plt.scatter(x, y, c=labels, edgecolors='k', cmap=plt.cm.RdYlBu, s=50) 
    plt.title("Decision Boundaries")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()
    plt.show()


def accuracy_fn(pred, label):
    val = torch.eq(pred, label).sum().item()
    acc = val / len


def conv2d_calc(input_dim: int, kernel, stride, padding):
    output = 1 + (input_dim + (padding * 2) - kernel) / stride

    return output


def layer_dim_calc(conv2d_count, maxpool_count, conv2d_kernel, maxpool_kernel, stride, padding):
    dim =0
    for epoch in range(0, maxpool_count, 1):
        for epoch in range(0, conv2d_count, 1):
            dim = conv2d_calc(dim, conv2d_kernel, stride, padding)
        dim /= maxpool_kernel
    return int(dim)

