# Installation of the necessary packages
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)

"""We use PyTorch in this part because it provides a highly flexible and efficient framework for building and training deep learning models, especially for tasks involving complex data structures like point clouds and graphs by providing the appropriate libraries such as pytorch geometric."""

!pip install -q torch-scatter -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-sparse -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}.html
!pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
!pip install -q wandb

!pip install torchmetrics

# Importation of Libraries
import os
import wandb
import random
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F

from torch_scatter import scatter
from torchmetrics.functional import jaccard_index

import torch_geometric.transforms as T
from torch_geometric.datasets import ShapeNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, DynamicEdgeConv

"""We use **Weights and Biases (W&B)** to efficiently track and visualize our model training, hyperparameters, and results in real-time. W&B provides a seamless way to log experiments, making it easy to compare different runs and configurations, which is especially helpful when experimenting with adversarial attacks and complex architectures like DGCNN.

We need to call [`wandb.init()`](https://docs.wandb.ai/ref/python/init) once at the beginning of our program to initialize a new job. This creates a new run in W&B and launches a background process to sync data.

To create your own project in W&B it's needed to get an API key from [`Get.My.API.Key`](https://wandb.ai/authorize) and enter it bellow.
"""

wandb_project = "pyg-point-cloud" #@param {"type": "string"}
wandb_run_name = "train-dgcnn" #@param {"type": "string"}

wandb.init(project=wandb_project, name=wandb_run_name, job_type="train")

config = wandb.config

config.seed = 42
config.device = 'cuda' if torch.cuda.is_available() else 'cpu'

random.seed(config.seed)
torch.manual_seed(config.seed)
device = torch.device(config.device)

config.category = 'Airplane' #@param ["Bag", "Cap", "Car", "Chair", "Earphone", "Guitar", "Knife", "Lamp", "Laptop", "Motorbike", "Mug", "Pistol", "Rocket", "Skateboard", "Table"] {type:"raw"}
config.random_jitter_translation = 1e-2
config.random_rotation_interval_x = 15
config.random_rotation_interval_y = 15
config.random_rotation_interval_z = 15
config.validation_split = 0.2
config.batch_size = 16
config.num_workers = 6

config.num_nearest_neighbours = 30
config.aggregation_operator = "max"
config.dropout = 0.5
config.initial_lr = 1e-3
config.lr_scheduler_step_size = 5
config.gamma = 0.8

config.epochs = 1

"""##1.2 Dataset Preparation

After we have prepared the necessary packages and dependencies, it's time to move to one of the essential steps: Dataset preparation.
"""

# Data augmentation transformations applied to point clouds.
# These transformations help improve the model's robustness by introducing random variations to the data.
transform = T.Compose([
    T.RandomJitter(config.random_jitter_translation),  # Adds random jitter to points for translation noise.
    T.RandomRotate(config.random_rotation_interval_x, axis=0),  # Randomly rotates the point cloud around the x-axis.
    T.RandomRotate(config.random_rotation_interval_y, axis=1),  # Randomly rotates the point cloud around the y-axis.
    T.RandomRotate(config.random_rotation_interval_z, axis=2)   # Randomly rotates the point cloud around the z-axis.
])

# Pre-processing step that normalizes the point cloud to fit within a unit sphere.
pre_transform = T.NormalizeScale()

"""**ShapeNet** is ideal for 3D object recognition due to its diverse, richly annotated dataset covering various object classes. It provides a standard benchmark for model comparison and supports training on realistic, real-world-like objects."""

#Downloading ShapeNet
dataset_path = os.path.join('ShapeNet', config.category)

train_val_dataset = ShapeNet(
    dataset_path, config.category, split='trainval',
    transform=transform, pre_transform=pre_transform
)

"""Now, we need to offset the segmentation labels"""

# Dictionary to store the frequency of each segmentation class label
segmentation_class_frequency = {}

# Iterate over the dataset to count the occurrence of each class label
for idx in tqdm(range(len(train_val_dataset))):
    # Extract point cloud and segmentation labels
    pc_viz = train_val_dataset[idx].pos.numpy().tolist()
    segmentation_label = train_val_dataset[idx].y.numpy().tolist()

    # Count the frequency of each class label
    for label in set(segmentation_label):
        segmentation_class_frequency[label] = segmentation_label.count(label)

# Find the minimum class label (used for normalization)
class_offset = min(list(segmentation_class_frequency.keys()))
print("Class Offset:", class_offset)

# Adjust the class labels by subtracting the minimum class label
for idx in range(len(train_val_dataset)):
    train_val_dataset[idx].y -= class_offset

"""Then, spliting data."""

num_train_examples = int((1 - config.validation_split) * len(train_val_dataset))
train_dataset = train_val_dataset[:num_train_examples]
val_dataset = train_val_dataset[num_train_examples:]

"""After that, we transform the splited data to DataLoaders for **Batch Processing**, **Shuffling**, **Parallel Loading** and **Data Augmentation**.

"""

train_loader = DataLoader(
    train_dataset, batch_size=config.batch_size,
    shuffle=True, num_workers=config.num_workers
)
val_loader = DataLoader(
    val_dataset, batch_size=config.batch_size,
    shuffle=False, num_workers=config.num_workers
)
visualization_loader = DataLoader(
    val_dataset[:10], batch_size=1,
    shuffle=False, num_workers=config.num_workers
)

"""##1.3 Implementation of the architecture

"""

class DGCNN(torch.nn.Module):
    def __init__(self, out_channels, k=30, aggr='max'):
        super().__init__()

        # Define the first Dynamic Edge Convolution layer with a Multi-Layer Perceptron (MLP) for feature transformation.
        # The input to this layer is a concatenation of point features and positions.
        # k specifies the number of nearest neighbors used in the convolution.
        # 'aggr' specifies the aggregation method for the convolution.
        self.conv1 = DynamicEdgeConv(MLP([2 * 6, 64, 64]), k, aggr)

        # Define the second Dynamic Edge Convolution layer with updated MLP for further feature transformation.
        # This layer takes the output from the first convolution layer as input.
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)

        # Define the third Dynamic Edge Convolution layer for more complex feature extraction.
        # The input size is adjusted according to the output of the second convolution layer.
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 64, 64]), k, aggr)

        # Define a Multi-Layer Perceptron (MLP) for classification.
        # This MLP processes the concatenated features from all three convolution layers.
        # dropout is applied for regularization to prevent overfitting.
        self.mlp = MLP(
            [3 * 64, 1024, 256, 128, out_channels],
            dropout=0.5, norm=None
        )

    def forward(self, data):
        # Extract input features, positions, and batch indices from the data object.
        x, pos, batch = data.x, data.pos, data.batch

        # Concatenate point features with positional information to form the input for the first convolution layer.
        x0 = torch.cat([x, pos], dim=-1)

        # Apply the first Dynamic Edge Convolution layer.
        x1 = self.conv1(x0, batch)

        # Apply the second Dynamic Edge Convolution layer to the output of the first layer.
        x2 = self.conv2(x1, batch)

        # Apply the third Dynamic Edge Convolution layer to the output of the second layer.
        x3 = self.conv3(x2, batch)

        # Concatenate the outputs from all three convolution layers and pass them through the MLP for classification.
        out = self.mlp(torch.cat([x1, x2, x3], dim=1))

        # Apply log_softmax to the output for classification.
        return F.log_softmax(out, dim=1)

# Set the number of output channels in the model to match the number of classes in the training dataset.
config.num_classes = train_dataset.num_classes

# Initialize the DGCNN model with the number of output channels (classes), the number of nearest neighbors, and the aggregation operator.
model = DGCNN(
    out_channels=train_dataset.num_classes,
    k=config.num_nearest_neighbours,
    aggr=config.aggregation_operator
).to(device)  # Move the model to the specified device (GPU/CPU).

# Initialize the Adam optimizer with the model parameters and a learning rate specified in the configuration.
optimizer = torch.optim.Adam(model.parameters(), lr=config.initial_lr)

# Set up a learning rate scheduler to adjust the learning rate during training.
# It reduces the learning rate by a factor of gamma every step_size epochs.
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=config.lr_scheduler_step_size, gamma=config.gamma
)

"""# 1.4 Training DGCNN"""

def train_step(epoch):
    # Set the model to training mode.
    model.train()

    ious, categories = [], []  # Lists to store IoU scores and categories.
    total_loss = correct_nodes = total_nodes = 0  # Initialize loss and accuracy counters.
    y_map = torch.empty(
        train_loader.dataset.num_classes, device=device
    ).long()  # Prepare a tensor for class mappings.
    num_train_examples = len(train_loader)  # Number of training batches.

    # Progress bar to track training progress.
    progress_bar = tqdm(
        train_loader, desc=f"Training Epoch {epoch}/{config.epochs}"
    )

    for data in progress_bar:
        data = data.to(device)  # Move data to the appropriate device (GPU/CPU).

        optimizer.zero_grad()  # Clear previous gradients.
        outs = model(data)  # Forward pass through the model.
        loss = F.nll_loss(outs, data.y)  # Compute the loss.
        loss.backward()  # Backward pass to compute gradients.
        optimizer.step()  # Update model weights.

        total_loss += loss.item()  # Accumulate total loss.
        correct_nodes += outs.argmax(dim=1).eq(data.y).sum().item()  # Count correct predictions.
        total_nodes += data.num_nodes  # Total number of nodes processed.

        sizes = (data.ptr[1:] - data.ptr[:-1]).tolist()  # Compute sizes of batches.
        for out, y, category in zip(outs.split(sizes), data.y.split(sizes),
                                    data.category.tolist()):
            category = list(ShapeNet.seg_classes.keys())[category]
            part = ShapeNet.seg_classes[category]
            part = torch.tensor(part, device=device)

            y_map[part] = torch.arange(part.size(0), device=device)

            # Compute Intersection over Union (IoU) for the segmentation.
            iou = jaccard_index(
                out[:, part].argmax(dim=-1), y_map[y],
                task="multiclass", num_classes=part.size(0)
            )
            ious.append(iou)  # Append IoU to the list.

        categories.append(data.category)  # Append categories for IoU calculation.

    # Compute mean IoU over all categories.
    iou = torch.tensor(ious, device=device)
    category = torch.cat(categories, dim=0)
    mean_iou = float(scatter(iou, category, reduce='mean').mean())

    # Return metrics for the current training epoch.
    return {
        "Train/Loss": total_loss / num_train_examples,
        "Train/Accuracy": correct_nodes / total_nodes,
        "Train/IoU": mean_iou
    }

"""This **`train_step`** function is responsible for executing a single training epoch for the model. It sets the model to training mode and processes each batch of data through forward and backward passes.

The function computes the loss using the negative log-likelihood loss function and updates the model parameters with the optimizer. It tracks the total loss, correct predictions, and overall accuracy across all nodes.

**Additionally**, it calculates **Intersection over Union (IoU)** scores for segmentation tasks, assessing the model's performance on different object categories. By averaging these metrics, the function provides a comprehensive evaluation of the training progress, including **loss**, **accuracy**, and **IoU**, for each epoch.

---

"""