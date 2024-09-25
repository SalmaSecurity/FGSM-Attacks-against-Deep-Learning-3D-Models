"""#**2. FGSM Attack Implementation**

In this section, we implement the Fast Gradient Sign Method (FGSM) attack, a well-known technique for generating adversarial examples in machine learning. FGSM perturbs the input data by taking a step in the direction of the gradient of the loss function with respect to the input. This method is used to evaluate the robustness of models by introducing small, targeted perturbations that can cause misclassification while remaining nearly imperceptible.

Key Steps:

*  Compute the Gradient: Calculate the gradient of the loss with respect to the input data.
*  Generate Perturbation: Modify the input data by adding a perturbation proportional to the gradient, scaled by a parameter called epsilon.
*  Update Data: Apply the perturbation to the original data to create adversarial examples.

##**2.1 Classic FGSM Attack**
"""

# Function to apply the FGSM attack
def fgsm_attack(model, images, labels, epsilon):
    """
    Applies the FGSM (Fast Gradient Sign Method) attack to generate adversarial examples.

    Parameters:
    - model: The model to be attacked.
    - images: The input images to attack.
    - labels: The true labels of the images.
    - epsilon: The magnitude of the perturbation for the attack.

    Returns:
    - adversarial_images: Images perturbed by the FGSM attack.
    """
    # Convert images and labels to TensorFlow tensors
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)

    # Enable gradient calculation
    with tf.GradientTape() as tape:
        tape.watch(images)  # Watch images for gradient calculation
        predictions = model(images)  # Obtain model predictions
        # Compute loss
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

    # Calculate the gradients of the loss with respect to the images
    gradients = tape.gradient(loss, images)
    # Compute perturbations using the sign of the gradients
    perturbations = epsilon * tf.sign(gradients)
    # Generate adversarial images by adding perturbations
    adversarial_images = images + perturbations

    return adversarial_images

# Set the magnitude of the perturbation for the FGSM attack
epsilon = 2 #Try with different values

# Retrieve all data and labels from the test dataset
all_samples = []
all_labels = []
for data in test_dataset:
    points, target = data
    all_samples.append(points)
    all_labels.append(target)

# Combine all samples and labels into single tensors
all_data = tf.concat(all_samples, axis=0)
all_labels = tf.concat(all_labels, axis=0)

# Set batch size and compute the number of batches
batch_size = 32  # Adjust this value based on GPU memory
num_batches = len(all_data) // batch_size + (1 if len(all_data) % batch_size != 0 else 0)

# Predictions before the attack
correct_before = 0
for i in range(num_batches):
    batch_data = all_data[i * batch_size: (i + 1) * batch_size]
    batch_labels = all_labels[i * batch_size: (i + 1) * batch_size]
    preds_before = model(batch_data)
    preds_before = tf.argmax(preds_before, axis=-1)
    correct_before += tf.reduce_sum(tf.cast(tf.equal(preds_before, batch_labels), tf.float32)).numpy()

# Apply the FGSM attack on the entire dataset
perturbed_data = []
for i in range(num_batches):
    batch_data = all_data[i * batch_size: (i + 1) * batch_size]
    perturbed_batch = fgsm_attack(model, batch_data, all_labels[i * batch_size: (i + 1) * batch_size], epsilon)
    perturbed_data.append(perturbed_batch)

# Combine perturbed batches into a single tensor
perturbed_data = tf.concat(perturbed_data, axis=0)

# Predictions after the attack
correct_after = 0
for i in range(num_batches):
    batch_data = perturbed_data[i * batch_size: (i + 1) * batch_size]
    batch_labels = all_labels[i * batch_size: (i + 1) * batch_size]
    preds_after = model(batch_data)
    preds_after = tf.argmax(preds_after, axis=-1)
    correct_after += tf.reduce_sum(tf.cast(tf.equal(preds_after, batch_labels), tf.float32)).numpy()

# Calculate accuracy
total_samples = len(all_labels)
accuracy_before = correct_before / total_samples * 100
accuracy_after = correct_after / total_samples * 100
attack_success_rate = (total_samples - correct_after) / total_samples * 100

# Display the results
print(f"Total number of samples: {total_samples}")
print(f"Number of correct predictions before the attack: {correct_before}")
print(f"Number of correct predictions after the attack: {correct_after}")
print(f"Accuracy before the attack: {accuracy_before:.2f}%")
print(f"Accuracy after the attack: {accuracy_after:.2f}%")
print(f"Attack success rate: {attack_success_rate:.2f}%")

"""We tested several values of epsilon to assess the impact of the FGSM attack on point clouds. The chosen values are [0.1, 0.25, 0.4, 0.5, 0.7, 1, 2]. Each epsilon value represents a different level of perturbation applied to the input data.

Here we give a visualization of differences between original Point clouds and perturbed ones
"""

import matplotlib.pyplot as plt

# Some samples
sample_indices = [0, 1, 2]  # To be visualized

original_samples = [all_data[i].numpy() for i in sample_indices]  # Conversion to numpy
perturbed_samples = [perturbed_data[i].numpy() for i in sample_indices]  # Conversion to numpy


"""Also we give here a smaller sample to test the attack. Especially, for 'sofa' data."""

import tensorflow as tf
import numpy as np

# 'Sofa' samples selection
sofa_index = next((key for key, value in CLASS_MAP.items() if value == 'sofa'), None)

if sofa_index is None:
    print("La classe 'sofa' n'a pas été trouvée dans CLASS_MAP.")
else:
    # Filtering
    sofa_samples = []
    for data in test_dataset:
        points, target = data
        mask = target == sofa_index
        sofa_samples.append((points[mask], target[mask]))

    if sofa_samples:
        sofa_data = tf.concat([x[0] for x in sofa_samples], axis=0)
        sofa_labels = tf.concat([x[1] for x in sofa_samples], axis=0)

        # Applying FGSM
        epsilon = 1  # Try different values
        perturbed_sofa_data = fgsm_attack(model, sofa_data, sofa_labels, epsilon)

        # Predictions before attack
        preds_before = model(sofa_data)
        preds_before = tf.argmax(preds_before, axis=-1)
        correct_before = tf.reduce_sum(tf.cast(tf.equal(preds_before, sofa_labels), tf.float32)).numpy()

        # Predictions after attack
        preds_after = model(perturbed_sofa_data)
        preds_after = tf.argmax(preds_after, axis=-1)
        correct_after = tf.reduce_sum(tf.cast(tf.equal(preds_after, sofa_labels), tf.float32)).numpy()

        # Results
        total_sofa_samples = len(sofa_labels)
        print(f"Total sofa samples: {total_sofa_samples}")
        print(f"Correct predictions before attack: {correct_before}")
        print(f"Correct predictions before attack: {correct_after}")
    else:
        print("None!")


