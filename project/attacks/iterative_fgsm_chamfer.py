##**2.2 FGSM Attack with Iterative Method**
"""
The following implementation extends the basic Fast Gradient Sign Method (FGSM) by applying the perturbation iteratively. In each iteration, a small perturbation is added to the input image in the direction of the gradient of the loss function, with the goal of maximizing the loss. We will use Chamfer Distance to control perturbations.
"""

import tensorflow as tf

# Function to apply the I-FGSM attack with Chamfer Distance
def ifgsm_attack_with_chamfer(model, images, labels, epsilon, chamfer_threshold, alpha, num_iterations):
    images = tf.convert_to_tensor(images)
    labels = tf.convert_to_tensor(labels)
    adversarial_images = tf.identity(images)

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            tape.watch(adversarial_images)
            predictions = model(adversarial_images)
            loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)

        gradients = tape.gradient(loss, adversarial_images)

        # Create perturbations
        perturbations = alpha * tf.sign(gradients)
        adversarial_images = adversarial_images + perturbations

        # Check the Chamfer Distance and adjust perturbations if necessary
        chamfer_dist = chamfer_distance(images, adversarial_images)
        if chamfer_dist > chamfer_threshold:
            scale_factor = chamfer_threshold / chamfer_dist
            adversarial_images = images + (adversarial_images - images) * scale_factor

        # Apply clipping to ensure the perturbations do not exceed epsilon
        perturbations_clipped = tf.clip_by_value(adversarial_images - images, -epsilon, epsilon)
        adversarial_images = images + perturbations_clipped

    return adversarial_images

# Parameters for the I-FGSM attack
epsilon = 10  # Set the maximum perturbation strength
alpha = 1.0  # Step size for each iteration
num_iterations = 15  # Number of iterations
chamfer_threshold = 0.5  # Adjust as needed

# Apply the I-FGSM attack on the entire dataset
perturbed_data = []
for i in range(num_batches):
    batch_data = all_data[i * batch_size: (i + 1) * batch_size]
    perturbed_batch = ifgsm_attack_with_chamfer(model, batch_data, all_labels[i * batch_size: (i + 1) * batch_size], epsilon, chamfer_threshold, alpha, num_iterations)
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

# Display results
print(f"Total number of samples: {total_samples}")
print(f"Number of correct predictions before the attack: {correct_before}")
print(f"Number of correct predictions after the attack: {correct_after}")
print(f"Accuracy before the attack: {accuracy_before:.2f}%")
print(f"Accuracy after the attack: {accuracy_after:.2f}%")
print(f"Attack success rate: {attack_success_rate:.2f}%")

