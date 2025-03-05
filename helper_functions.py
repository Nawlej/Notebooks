import torch
import numpy as np
import matplotlib.pyplot as plt

def shuffle(input: torch.Tensor, seed: int =None):
    if seed != None: 
        torch.manual_seed(seed)
    idx = torch.randperm(input.shape[0])
    
    return input.view(input.shape)[idx]


def plot_decision_boundaries(model: torch.nn.Module, data: torch.Tensor, labels: torch.Tensor):
    """ Plots the decision boundary in a 2D space
    """
    data = data.detach().numpy()
    data_x = data[:, 0]
    data_y = data[:, 1]

    x_min, x_max = data_x.min() - 1, data_x.max() + 1
    y_min, y_max = data_y.min() - 1, data_y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Setup grid points and predict the model's output for said grid points
    grid_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

    model.eval()
    with torch.inference_mode():
        logits = model(grid_points).squeeze()

        # check for multiclassification 
        if len(logits.shape) > 1:
            logits = logits.argmax(1)
        else:
            logits = logits.sigmoid()
        logits = logits.reshape(xx.shape)
        
    # Plot the decision boundary
    plt.contourf(xx, yy, logits, cmap=plt.cm.RdYlBu, alpha=0.7) 
    plt.scatter(data_x, data_y, c=labels, edgecolors='k', cmap=plt.cm.RdYlBu, s=50) 
    plt.title("Decision Boundaries")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar()




