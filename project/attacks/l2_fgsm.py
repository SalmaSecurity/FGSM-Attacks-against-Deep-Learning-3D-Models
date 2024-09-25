"""##**2.3 FGSM Attack with L2 Norm**

FGSM is designed to create perturbations that fools the model by making small, intentional changes to the input data. When using FGSM with L2 norm, the perturbations are constrained by the L2 distance metric, which measures the "straight-line" distance between the original and perturbed inputs.
"""

import tensorflow as tf
import numpy as np

# Function to apply FGSM attack
def fgsm_attack(model, images, labels, epsilon):
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)

    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

    gradients = tape.gradient(loss, images)
    l2_norm = tf.norm(gradients)
    normalized_gradients = gradients / (l2_norm + 1e-10)

    # Create perturbed images
    perturbations = epsilon * normalized_gradients  # Apply normalized gradients
    adversarial_images = images + perturbations

    return adversarial_images

# Apply the attack on the entire dataset
epsilon = 100  # Define the perturbation strength

# Retrieve all data and labels from the dataset
all_samples = []
all_labels = []
for data in test_dataset:
    points, target = data
    all_samples.append(points)
    all_labels.append(target)

# Combine samples
all_data = tf.concat(all_samples, axis=0)
all_labels = tf.concat(all_labels, axis=0)

# Batch size parameter
batch_size = 32  # Adjust this value according to your GPU memory
num_batches = len(all_data) // batch_size + (1 if len(all_data) % batch_size != 0 else 0)

# Predictions before the attack
correct_before = 0
for i in range(num_batches):
    batch_data = all_data[i * batch_size: (i + 1) * batch_size]
    batch_labels = all_labels[i * batch_size: (i + 1) * batch_size]
    preds_before = model(batch_data)
    preds_before = tf.argmax(preds_before, axis=-1)
    correct_before += tf.reduce_sum(tf.cast(tf.equal(preds_before, batch_labels), tf.float32)).numpy()

# Apply FGSM attack on the entire dataset
perturbed_data = []
for i in range(num_batches):
    batch_data = all_data[i * batch_size: (i + 1) * batch_size]
    perturbed_batch = fgsm_attack(model, batch_data, all_labels[i * batch_size: (i + 1) * batch_size], epsilon)
    perturbed_data.append(perturbed_batch)

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

# Print results
print(f"Total number of samples: {total_samples}")
print(f"Number of correct predictions before the attack: {correct_before}")
print(f"Number of correct predictions after the attack: {correct_after}")
print(f"Accuracy before the attack: {accuracy_before:.2f}%")
print(f"Accuracy after the attack: {accuracy_after:.2f}%")
print(f"Attack success rate: {attack_success_rate:.2f}%")
