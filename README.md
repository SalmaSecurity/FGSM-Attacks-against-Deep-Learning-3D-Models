# FGSM-Attacks-against-Deep-Learning-3D-Models

This project demonstrates how FGSM (Fast Gradient Sign Method) attacks can be applied to deep learning models trained on 3D datasets. We explore multiple implementations of FGSM on two key models: PointNet and DGCNN, using different distance metrics like L2 and Chamfer distances.

## What are Adversarial Attacks ?

Adversarial attacks are techniques used to disrupt the performance of machine learning models by introducing subtle perturbations into the input data. These attacks can render a model vulnerable by causing it to make incorrect predictions, even though the perturbations are imperceptible to humans. The severity of these attacks is significant, as they can compromise the security of intelligent systems in critical applications such as image recognition, autonomous driving, or biometrics.

## What is Point Cloud and Its Importance

A point cloud is a collection of three-dimensional points representing the surface of an object or scene. This format is commonly used in computer vision and 3D modeling for tasks such as object recognition, 3D reconstruction, and autonomous navigation. The importance of point clouds lies in their ability to capture precise geometric details, which is crucial for models that require a fine spatial understanding.

#How do you capture them?

There are basically two methods to generate point cloud.


### 1.   **Laser Scanner** :
It uses high speed laser pulses to accumulate high density accurate measurements. And also include RGB-D camera to add color and depth information.


### 2.   **Photogrammetry** :
It is a method to create a 3D reconstructed using cameras. The cameras capture the spaces from all angles and then process using softwares like Agisoft Metashape, AliceVision Meshroom to create a 3D image in space.



## What are PointNet and FGSM Attack

PointNet is a pioneering model for processing point clouds, capable of extracting discriminative features from 3D data. However, like many other machine learning models, PointNet is vulnerable to adversarial attacks. One of the simplest and most effective attack methods is the Fast Gradient Sign Method (FGSM). This attack involves adding a calculated perturbation to maximize the model’s loss while remaining almost imperceptible.

## Dynamic Graph Convolutional Neural Network (**DGCNN**)

DGCNN is a neural network architecture designed to handle point cloud data by using the structure of graphs.
Unlike traditional Convolutional Neural Networks (CNNs), which are suited for regular grid-like data such as images, DGCNN is
built to process non-Euclidean data like 3D point clouds. This makes it an essential tool in various applications involving
3D object recognition, scene segmentation, and point cloud classification.

## How DGCNN Works

The process begins by constructing a graph where points are nodes, and edges represent the relationships between neighboring
points based on distance metrics. During the training process, the model dynamically updates these graphs at each layer of the
network, allowing for a deeper understanding of the underlying 3D structure. The use of edge features in convolutional operations
provides a way to capture fine geometric details that are crucial for accurate classification
and segmentation tasks.

## Adversarial Attacks on DGCNN

Adversarial attacks are a well-known challenge in deep learning, particularly in fields like computer vision and natural language
processing. These attacks involve perturbing input data to deceive neural networks into making incorrect predictions.
Recently, these types of attacks have been extended to 3D point cloud models like DGCNN (Dynamic Graph Convolutional Neural Network),
which presents new complexities due to the non-Euclidean nature of point cloud data.

## Features
- Implementation of PointNet and DGCNN models for 3D point cloud classification/segmentation.
- FGSM adversarial attacks with various distance metrics:
  - Classic FGSM.
  - FGSM with L2 norm.
  - FGSM with Chamfer distance.
  - Iterative FGSM with Chamfer distance.
- Visualization of the impact and surface state analysis through roughness.

## Project Structure
project/

├── models/

│   ├── dgcnn.py        # Implementation of DGCNN

│   ├── pointnet.py     # Implementation of PointNet

├── attacks/

│   ├── basic_fgsm.py             # Basic FGSM attack 

│   ├── chamfer_fgsm.py           # FGSM attack with Chamfer distance

│   ├── iterative_fgsm_chamfer    # Iterative FGSM attack with Chamfer distance

│   ├── l2_fgsm                   # FGSM attack with l2 norm 

│   ├── l2_fgsm_dgcnn             # FGSM attack with l2 norm (DGCNN)

├── Visualization_Roughness     

├── README.md            


## Usage

It is recommended to execute the code in **Google Colab** or **Kaggle** to take advantage of GPU acceleration and pre-installed libraries. This also helps in managing outputs and the details of the execution process efficiently.

### Execution Order
To run the code, please follow this execution order:

1. **Implementation of Models**
   - Start by running the implementations of the PointNet and DGCNN models.

2. **Adversarial Attacks**
   - Execute the FGSM attacks in the following sequence:
     - FGSM with Chamfer distance
     - FGSM with Iterative Chamfer distance
     - FGSM with L2 norm

3. **Visualization**
   - After running the attacks, visualize the results to understand the impact of adversarial examples on the models.

4. **Roughness Calculation**
   - Finally, calculate the roughness of the generated adversarial examples to assess their quality and effectiveness.




