"""Let's visualize the impact on point clouds"""

import matplotlib.pyplot as plt

# Choose three sample indices (make sure they are valid)
sample_indices = [0, 1, 2]  # Indices of samples to visualize

# Original and perturbed objects
# If all_data is a tensor with shape (N, num_points, num_features), extract samples as follows:
original_samples = [all_data[i].numpy() for i in sample_indices]  # Convert to numpy
perturbed_samples = [perturbed_data[i].numpy() for i in sample_indices]  # Convert to numpy

# Visualization
fig = plt.figure(figsize=(12, 6))

for i, idx in enumerate(sample_indices):
    # Subplot for the original sample
    ax1 = fig.add_subplot(2, 3, i + 1, projection='3d')
    ax1.scatter(original_samples[i][:, 0], original_samples[i][:, 1], original_samples[i][:, 2], c='blue', s=5)
    ax1.set_title(f'Original Sample {idx}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Subplot for the perturbed sample
    ax2 = fig.add_subplot(2, 3, i + 4, projection='3d')
    ax2.scatter(perturbed_samples[i][:, 0], perturbed_samples[i][:, 1], perturbed_samples[i][:, 2], c='red', s=5)
    ax2.set_title(f'Perturbed Sample {idx}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

plt.tight_layout()
plt.show()

"""##**Roughness Calculation for this Attack with L2 Norm**

Using always the same method, but this time we will test with 'bed' object
"""

import open3d as o3d
import numpy as np
import tensorflow as tf
from google.colab import files

# Fixed indices for the 'bed' class (2)
bed_indices = [i for i, label in enumerate(all_labels) if label == 2]

# Function to transform point clouds into Open3D PointCloud objects
def transform_point_cloud(point_cloud):
    if isinstance(point_cloud, tf.Tensor):
        point_cloud = point_cloud.numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    if point_cloud.shape[1] > 3:
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255.0)

    if point_cloud.shape[1] > 6:
        pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 6:9])
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    return pcd

# Function to calculate roughness using the Lavoué method
def calculate_lavoue_roughness(mesh, k_neighbors=10):
    vertices = np.asarray(mesh.vertices)
    kd_tree = o3d.geometry.KDTreeFlann(mesh)
    roughness = np.zeros(len(vertices))

    for i in range(len(vertices)):
        [_, idx, _] = kd_tree.search_knn_vector_3d(vertices[i], k_neighbors)
        neighbors = np.asarray([vertices[j] for j in idx[1:]])  # Exclude the point itself
        distances = np.linalg.norm(neighbors - vertices[i], axis=1)
        roughness[i] = np.var(distances)

    return np.mean(roughness)

# List of epsilon values to test
epsilon_values = [0.1, 0.5, 1, 2]

# Reduce batch size to avoid memory exhaustion
batch_size = 10  # Adjust according to your GPU capabilities

# Function to handle GPU memory
def setup_gpu_memory():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(f"Error in setting memory growth: {e}")

setup_gpu_memory()

for epsilon in epsilon_values:
    perturbed_data = []  # Reset list of perturbed data for each epsilon value

    for i in range(0, len(all_data), batch_size):
        batch_data = all_data[i: i + batch_size]
        batch_labels = all_labels[i: i + batch_size]

        # Apply FGSM attack
        perturbed_batch = fgsm_attack(model, batch_data, batch_labels, epsilon)
        perturbed_data.append(perturbed_batch)

    perturbed_data = tf.concat(perturbed_data, axis=0)  # Concatenate all perturbed data

    original_samples = [all_data[i].numpy() for i in bed_indices[:2]]  # Two 'bed' objects
    perturbed_samples = [perturbed_data[i].numpy() for i in bed_indices[:2]]

    for i in range(len(original_samples)):
        original_pcd = transform_point_cloud(original_samples[i])
        perturbed_pcd = transform_point_cloud(perturbed_samples[i])

        # Calculate average distance of neighbors to determine radius
        distances = original_pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        # Create meshes from point clouds
        original_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            original_pcd, o3d.utility.DoubleVector([radius, radius * 2]))

        perturbed_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            perturbed_pcd, o3d.utility.DoubleVector([radius, radius * 2]))

        # Simplify meshes
        original_mesh = original_mesh.simplify_quadric_decimation(100000)
        perturbed_mesh = perturbed_mesh.simplify_quadric_decimation(100000)

        # Calculate roughness
        original_roughness = calculate_lavoue_roughness(original_mesh)
        perturbed_roughness = calculate_lavoue_roughness(perturbed_mesh)

        print(f"epsilon: {epsilon}, Object: Bed {i+1}")
        print(f"Original Roughness: {original_roughness}")
        print(f"Perturbed Roughness: {perturbed_roughness}\n")

        # Save meshes as PLY files
        original_mesh_path = f"original_mesh_epsilon_{epsilon}_bed_{i+1}.ply"
        perturbed_mesh_path = f"perturbed_mesh_epsilon_{epsilon}_bed_{i+1}.ply"
        o3d.io.write_triangle_mesh(original_mesh_path, original_mesh)
        o3d.io.write_triangle_mesh(perturbed_mesh_path, perturbed_mesh)
        print(f"Original mesh saved to: {original_mesh_path}")
        print(f"Perturbed mesh saved to: {perturbed_mesh_path}")

        # Download PLY files
        files.download(original_mesh_path)
        files.download(perturbed_mesh_path)

"""Bellow we use the maximal values of roughness instead of the average"""

import open3d as o3d
import numpy as np
import tensorflow as tf
from google.colab import files

# Fixed indices for the 'bed' class (2)
bed_indices = [i for i, label in enumerate(all_labels) if label == 2]

# Function to transform point clouds into Open3D PointCloud objects
def transform_point_cloud(point_cloud):
    if isinstance(point_cloud, tf.Tensor):
        point_cloud = point_cloud.numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    if point_cloud.shape[1] > 3:
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255.0)

    if point_cloud.shape[1] > 6:
        pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 6:9])
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    return pcd

# Function to calculate roughness using the Lavoué method
def calculate_lavoue_roughness(mesh, k_neighbors=10):
    vertices = np.asarray(mesh.vertices)
    kd_tree = o3d.geometry.KDTreeFlann(mesh)
    roughness = np.zeros(len(vertices))

    for i in range(len(vertices)):
        [_, idx, _] = kd_tree.search_knn_vector_3d(vertices[i], k_neighbors)
        neighbors = np.asarray([vertices[j] for j in idx[1:]])  # Exclude the point itself
        distances = np.linalg.norm(neighbors - vertices[i], axis=1)
        roughness[i] = np.var(distances)

    return np.max(roughness)

# List of epsilon values to test
epsilon_values = [0.1, 0.5, 1, 2]

# Reduce batch size to avoid memory exhaustion
batch_size = 10  # Adjust according to your GPU capabilities

# Function to handle GPU memory
def setup_gpu_memory():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(f"Error in setting memory growth: {e}")

setup_gpu_memory()

for epsilon in epsilon_values:
    perturbed_data = []  # Reset list of perturbed data for each epsilon value

    for i in range(0, len(all_data), batch_size):
        batch_data = all_data[i: i + batch_size]
        batch_labels = all_labels[i: i + batch_size]

        # Apply FGSM attack
        perturbed_batch = fgsm_attack(model, batch_data, batch_labels, epsilon)
        perturbed_data.append(perturbed_batch)

    perturbed_data = tf.concat(perturbed_data, axis=0)  # Concatenate all perturbed data

    original_samples = [all_data[i].numpy() for i in bed_indices[:2]]  # Two 'bed' objects
    perturbed_samples = [perturbed_data[i].numpy() for i in bed_indices[:2]]

    for i in range(len(original_samples)):
        original_pcd = transform_point_cloud(original_samples[i])
        perturbed_pcd = transform_point_cloud(perturbed_samples[i])

        # Calculate average distance of neighbors to determine radius
        distances = original_pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        # Create meshes from point clouds
        original_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            original_pcd, o3d.utility.DoubleVector([radius, radius * 2]))

        perturbed_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            perturbed_pcd, o3d.utility.DoubleVector([radius, radius * 2]))

        # Simplify meshes
        original_mesh = original_mesh.simplify_quadric_decimation(100000)
        perturbed_mesh = perturbed_mesh.simplify_quadric_decimation(100000)

        # Calculate roughness
        original_roughness = calculate_lavoue_roughness(original_mesh)
        perturbed_roughness = calculate_lavoue_roughness(perturbed_mesh)

        print(f"epsilon: {epsilon}, Object: Bed {i+1}")
        print(f"Original Roughness: {original_roughness}")
        print(f"Perturbed Roughness: {perturbed_roughness}\n")

        # Save meshes as PLY files
        original_mesh_path = f"original_mesh_epsilon_{epsilon}_bed_{i+1}.ply"
        perturbed_mesh_path = f"perturbed_mesh_epsilon_{epsilon}_bed_{i+1}.ply"
        o3d.io.write_triangle_mesh(original_mesh_path, original_mesh)
        o3d.io.write_triangle_mesh(perturbed_mesh_path, perturbed_mesh)
        print(f"Original mesh saved to: {original_mesh_path}")
        print(f"Perturbed mesh saved to: {perturbed_mesh_path}")

        # Download PLY files
        files.download(original_mesh_path)
        files.download(perturbed_mesh_path)

"""Here we present the differences also"""

import open3d as o3d
import numpy as np
import tensorflow as tf
from google.colab import files

# Fixed indices for the 'bed' class (2)
bed_indices = [i for i, label in enumerate(all_labels) if label == 2]

# Function to transform point clouds into Open3D PointCloud objects
def transform_point_cloud(point_cloud):
    if isinstance(point_cloud, tf.Tensor):
        point_cloud = point_cloud.numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])

    if point_cloud.shape[1] > 3:
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:6] / 255.0)

    if point_cloud.shape[1] > 6:
        pcd.normals = o3d.utility.Vector3dVector(point_cloud[:, 6:9])
    else:
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    return pcd

# Function to calculate roughness using the Lavoué method
def calculate_lavoue_roughness(mesh, k_neighbors=10):
    vertices = np.asarray(mesh.vertices)
    kd_tree = o3d.geometry.KDTreeFlann(mesh)
    roughness = np.zeros(len(vertices))

    for i in range(len(vertices)):
        [_, idx, _] = kd_tree.search_knn_vector_3d(vertices[i], k_neighbors)
        neighbors = np.asarray([vertices[j] for j in idx[1:]])  # Exclude the point itself
        distances = np.linalg.norm(neighbors - vertices[i], axis=1)
        roughness[i] = np.var(distances)

    return np.mean(roughness)  # Adjust if needed to np.max(roughness)

# List of epsilon values to test
epsilon_values = [0.1, 0.5, 1, 2]

# Reduce batch size to avoid memory exhaustion
batch_size = 10  # Adjust according to your GPU capabilities

# Function to handle GPU memory
def setup_gpu_memory():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        except Exception as e:
            print(f"Error in setting memory growth: {e}")

setup_gpu_memory()

for epsilon in epsilon_values:
    perturbed_data = []  # Reset list of perturbed data for each epsilon value

    for i in range(0, len(all_data), batch_size):
        batch_data = all_data[i: i + batch_size]
        batch_labels = all_labels[i: i + batch_size]

        # Apply FGSM attack
        perturbed_batch = fgsm_attack(model, batch_data, batch_labels, epsilon)
        perturbed_data.append(perturbed_batch)

    perturbed_data = tf.concat(perturbed_data, axis=0)  # Concatenate all perturbed data

    original_samples = [all_data[i].numpy() for i in bed_indices[:2]]  # Two 'bed' objects
    perturbed_samples = [perturbed_data[i].numpy() for i in bed_indices[:2]]

    for i in range(len(original_samples)):
        original_pcd = transform_point_cloud(original_samples[i])
        perturbed_pcd = transform_point_cloud(perturbed_samples[i])

        # Calculate average distance of neighbors to determine radius
        distances = original_pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        # Create meshes from point clouds
        original_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            original_pcd, o3d.utility.DoubleVector([radius, radius * 2]))

        perturbed_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            perturbed_pcd, o3d.utility.DoubleVector([radius, radius * 2]))

        # Simplify meshes
        original_mesh = original_mesh.simplify_quadric_decimation(100000)
        perturbed_mesh = perturbed_mesh.simplify_quadric_decimation(100000)

        # Calculate roughness
        original_roughness = calculate_lavoue_roughness(original_mesh)
        perturbed_roughness = calculate_lavoue_roughness(perturbed_mesh)

        # Calculate roughness difference
        roughness_diff = perturbed_roughness - original_roughness

        print(f"epsilon: {epsilon}, Object: Bed {i+1}")
        print(f"Original Roughness: {original_roughness}")
        print(f"Perturbed Roughness: {perturbed_roughness}")
        print(f"Roughness Difference: {roughness_diff}\n")

        # Save meshes as PLY files
        original_mesh_path = f"original_mesh_epsilon_{epsilon}_bed_{i+1}.ply"
        perturbed_mesh_path = f"perturbed_mesh_epsilon_{epsilon}_bed_{i+1}.ply"
        o3d.io.write_triangle_mesh(original_mesh_path, original_mesh)
        o3d.io.write_triangle_mesh(perturbed_mesh_path, perturbed_mesh)
        print(f"Original mesh saved to: {original_mesh_path}")
        print(f"Perturbed mesh saved to: {perturbed_mesh_path}")

        # Download PLY files
        files.download(original_mesh_path)
        files.download(perturbed_mesh_path)