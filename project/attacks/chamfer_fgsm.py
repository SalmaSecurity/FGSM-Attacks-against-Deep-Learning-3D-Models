##**2.1 FGSM Attack with Chamfer Distance**
"""
This section demonstrates the implementation of the Fast Gradient Sign Method (FGSM) attack using Chamfer Distance as the evaluation metric. We will see how could this optimize the attack.

Applying FGSM attack to images and adjusts the perturbations based on Chamfer distance.

    Parameters:
    - model: The model to be attacked.
    - images: The input images to attack.
    - labels: The true labels of the images.
    - epsilon: The magnitude of the perturbation for the attack.
    - chamfer_threshold: The maximum allowed Chamfer distance for perturbations.

    Returns:
    - adversarial_images: Adjusted adversarial images after applying the attack.
"""

import tensorflow as tf
import numpy as np

# Definition of  Chamfer distance
def chamfer_distance(P, Q):
    dist = tf.reduce_mean(tf.reduce_min(tf.norm(tf.expand_dims(Q, 1) - tf.expand_dims(P, 0), axis=-1), axis=1))

    return dist

# FGSM
def fgsm_attack_with_chamfer(model, images, labels, epsilon, chamfer_threshold):
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)

    with tf.GradientTape() as tape:
        tape.watch(images)
        predictions = model(images)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

    gradients = tape.gradient(loss, images)

    perturbations = epsilon * tf.sign(gradients)
    adversarial_images = images + perturbations

    chamfer_dist = chamfer_distance(images, adversarial_images)

    if chamfer_dist > chamfer_threshold:
        scale_factor = chamfer_threshold / chamfer_dist
        adversarial_images = images + perturbations * scale_factor

    return adversarial_images

epsilon = 4  # Strenght of the pertubation
chamfer_threshold = 1  # We test with multiple values

all_samples = []
all_labels = []
for data in test_dataset:
    points, target = data
    all_samples.append(points)
    all_labels.append(target)

all_data = tf.concat(all_samples, axis=0)
all_labels = tf.concat(all_labels, axis=0)

batch_size = 32
num_batches = len(all_data) // batch_size + (1 if len(all_data) % batch_size != 0 else 0)

# Predictions before the attack
correct_before = 0
for i in range(num_batches):
    batch_data = all_data[i * batch_size: (i + 1) * batch_size]
    batch_labels = all_labels[i * batch_size: (i + 1) * batch_size]
    preds_before = model(batch_data)
    preds_before = tf.argmax(preds_before, axis=-1)
    correct_before += tf.reduce_sum(tf.cast(tf.equal(preds_before, batch_labels), tf.float32)).numpy()

# Application of FGSM attack
perturbed_data = []
for i in range(num_batches):
    batch_data = all_data[i * batch_size: (i + 1) * batch_size]
    perturbed_batch = fgsm_attack_with_chamfer(model, batch_data, all_labels[i * batch_size: (i + 1) * batch_size], epsilon, chamfer_threshold)
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

# Accuracy
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

