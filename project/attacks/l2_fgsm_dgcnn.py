

"""#**2. FGSM Attack using l2 norm**

The FGSM (Fast Gradient Sign Method) L2 attack generates adversarial examples by perturbing the input data to mislead the model. First, the model is set to evaluation mode and the data tensors, including node features, true labels, and graph connections, are prepared. Gradients are computed for the input data, and perturbations are scaled by a factor called `epsilon`. These perturbations are added to the original features to create a perturbed dataset. This approach evaluates how slight modifications can affect the model’s performance, highlighting its vulnerability to adversarial attacks.
"""

import torch
from torch_geometric.data import Data, DataLoader

def fgsm_l2_attack(model, data, epsilon):
    perturbed_data_list = []
    model.eval()
    data = data.to(device)
    data.x.requires_grad = True  # Ensure gradients can be computed for the data

    # Obtain predictions
    output = model(data)
    loss = F.nll_loss(output, data.y)
    model.zero_grad()
    loss.backward()

    # Compute perturbations
    perturbation = epsilon * data.x.grad / data.x.grad.norm(p=2, dim=1, keepdim=True)
    perturbed_x = data.x + perturbation

    # Create a new data object with perturbations
    perturbed_data = Data(x=perturbed_x, edge_index=data.edge_index, y=data.y, pos=data.pos, batch=data.batch)
    perturbed_data_list.append(perturbed_data)
    perturbed_dataset = perturbed_data_list

    return perturbed_dataset

"""We add a part of the code to evaluate the performance of the model after attack."""

def evaluate_attack(model, perturbed_data):
    model.eval()
    correct_nodes = total_nodes = 0

    with torch.no_grad():
        for data in perturbed_data:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct_nodes += pred.eq(data.y).sum().item()
            total_nodes += data.num_nodes

    accuracy = correct_nodes / total_nodes
    return accuracy

import wandb

# Initialize W&B
wandb.init(project="FGSM Attack")

# Apply FGSM L2 attack and evaluate
epsilon = 2  # Epsilon value for the FGSM attack

for data in val_loader:
    data = data.to(device)

    # Apply FGSM L2 attack
    perturbed_data = fgsm_l2_attack(model, data, epsilon)

    # Create a DataLoader for the perturbed data if needed
    #train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    #perturbed_loader = DataLoader(perturbed_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    # Evaluate the adversarial data
    accuracy = evaluate_attack(model, perturbed_data)

    # Log the results
    wandb.log({"Adversarial Accuracy": accuracy})

# Finish tracking with W&B
wandb.finish()

import matplotlib.pyplot as plt
import torch
from torch_geometric.data import Data

def visualize_point_cloud(pc, title="Point Cloud"):
    """Visualize a point cloud in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    ax.set_title(title)
    plt.show()

def visualize_difference(original, perturbed, title="Difference Visualization"):
    """Visualize the difference between the original and perturbed point clouds."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(original[:, 0], original[:, 1], original[:, 2], color='blue', label='Original', s=1)
    ax.scatter(perturbed[:, 0], perturbed[:, 1], perturbed[:, 2], color='red', label='Perturbed', s=1)
    ax.set_title(title)
    plt.legend()
    plt.show()

# Example usage
epsilon = 100
sample = train_val_dataset[0]  # Make sure train_val_dataset is correctly defined
perturbed_data = fgsm_l2_attack(model, sample, epsilon)  # Make sure fgsm_l2_attack is correctly defined

# Visualize the original point cloud
visualize_point_cloud(sample.pos.detach().cpu().numpy(), title="Original Point Cloud")

# Visualize the perturbed point cloud
visualize_point_cloud(perturbed_data.pos.detach().cpu().numpy(), title="Perturbed Point Cloud")

# Visualize the difference
visualize_difference(sample.pos.detach().cpu().numpy(), perturbed_data.pos.detach().cpu().numpy(), title="Original vs Perturbed Point Cloud")

import matplotlib.pyplot as plt

def visualize_point_cloud(pc, title="Point Cloud"):
    """Visualize a point cloud in 3D."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], s=1)
    ax.set_title(title)
    plt.show()

# Visualize a sample before transformation
sample = train_val_dataset[0]
epsilon = 10

# Apply the FGSM attack to obtain perturbed data
perturbed_data = fgsm_l2_attack(model, sample, epsilon)

# Visualize the original point cloud
visualize_point_cloud(sample.pos.cpu().numpy(), title="Original Point Cloud")

# Visualize the perturbed point cloud
visualize_point_cloud(perturbed_data.pos.cpu().numpy(), title="Perturbed Point Cloud")

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.data import Data, DataLoader

def fgsm_l2_attack(model, data, epsilon):
    model.eval()
    data = data.to(device)
    data.x.requires_grad = True  # Ensure gradients can be calculated for the data

    # Obtain predictions
    output = model(data)
    loss = F.nll_loss(output, data.y)
    model.zero_grad()
    loss.backward()

    # Calculate perturbations
    perturbation = epsilon * data.x.grad / data.x.grad.norm(p=2, dim=1, keepdim=True)
    perturbed_x = data.x + perturbation

    # Create a new data object with the perturbations
    perturbed_data = Data(x=perturbed_x, edge_index=data.edge_index, y=data.y, pos=data.pos, batch=data.batch)

    return data, perturbed_data

def visualize_comparison(original_data, perturbed_data):
    # Compare features of the two datasets
    original_x = original_data.x.cpu().detach().numpy()
    perturbed_x = perturbed_data.x.cpu().detach().numpy()

    # Visualization
    plt.figure(figsize=(12, 6))

    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Original Image Features")
    plt.imshow(original_x, cmap='gray')
    plt.colorbar()

    # Perturbed image
    plt.subplot(1, 2, 2)
    plt.title("Perturbed Image Features")
    plt.imshow(perturbed_x, cmap='gray')
    plt.colorbar()

    plt.show()

# Assuming 'model' and 'data' are already defined
original_data, perturbed_data = fgsm_l2_attack(model, data, epsilon=0.1)
visualize_comparison(original_data, perturbed_data)