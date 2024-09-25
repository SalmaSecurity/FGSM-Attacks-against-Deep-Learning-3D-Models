!pip install trimesh

#Then we import some needed libraries
import os
import glob
import trimesh
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
# Setting a random seed for TensorFlow in order to ensure reproducibility of results
tf.random.set_seed(1234)

"""##**1.1 Data Preparation**"""

#Downloading data
DATA_DIR = tf.keras.utils.get_file(
    "modelnet.zip",
    "http://3dvision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip",
    extract=True,
)
DATA_DIR = os.path.join(os.path.dirname(DATA_DIR), "ModelNet10")

"""**TensorFlow**

TensorFlow was selected for its efficiency in managing and optimizing machine learning models, particularly for implementing FGSM (Fast Gradient Sign Method) attacks. Its seamless integration with neural networks makes it ideal for applying adversarial perturbations and evaluating their impact on classification models.

**Trimesh**

Trimesh was chosen for its robust capabilities in handling, manipulating, and analyzing 3D meshes. It provides essential functions for mesh simplification, geometric property calculation, and assessing mesh quality, which are crucial for evaluating the effects of adversarial attacks on 3D point clouds.

Object File Format
"""

mesh = trimesh.load(os.path.join(DATA_DIR, "chair/train/chair_0001.off"))
mesh.show()

#After downloading mesh data, now we sample 2048 points from the 3D mesh surface to visualizing them using a 3D scatter plot

points = mesh.sample(2048)

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2])
ax.set_axis_off()
plt.show()

"""The following function parses the dataset to create training and testing sets by sampling points from 3D meshes.

    Parameters:
    - num_points: Presents the number of points to sample from each mesh (2048).

    Returns:
    - train_points, test_points: Arrays of sampled points from training and testing meshes.
    - train_labels, test_labels: Corresponding labels for training and testing points.
    - class_map: Dictionary mapping class indices (0,1,2...) to class names (sofa, tables...).


"""

def parse_dataset(num_points=2048):

    #We initialize lists for storing sampled points and labels for training and testing.
    train_points = []
    train_labels = []
    test_points = []
    test_labels = []

    #Create a class map from folder names in DATA_DIR, excluding any README files.
    class_map = {}
    folders = glob.glob(os.path.join(DATA_DIR, "[!README]*"))


    #For each class folder, load and sample points from training and testing meshes, storing points and labels.
    for i, folder in enumerate(folders):
        print("processing class: {}".format(os.path.basename(folder)))
        # store folder name with ID so we can retrieve later
        class_map[i] = folder.split("/")[-1]
        # gather all files
        train_files = glob.glob(os.path.join(folder, "train/*"))
        test_files = glob.glob(os.path.join(folder, "test/*"))

        for f in train_files:
            train_points.append(trimesh.load(f).sample(num_points))
            train_labels.append(i)

        for f in test_files:
            test_points.append(trimesh.load(f).sample(num_points))
            test_labels.append(i)

    #Convert lists to numpy arrays and return them along with the class map.
    return (
        np.array(train_points),
        np.array(test_points),
        np.array(train_labels),
        np.array(test_labels),
        class_map,
    )

#It's time to define constants for the dataset
NUM_POINTS = 2048       # Number of points to sample from each 3D mesh
NUM_CLASSES = 10        # Total number of classes in the dataset
BATCH_SIZE = 32         # Batch size for training

# Parsing the dataset to obtain training and testing points and labels, along with a class mapping (This takes up to 5 min)
train_points, test_points, train_labels, test_labels, CLASS_MAP = parse_dataset(NUM_POINTS)

"""This function aims to augment the point cloud data by adding random jitter and shuffling the points.

    Parameters:
    - points: A tensor representing the 3D points of a mesh.
    - label: The class label associated with the points.

    Returns:
    - Augmented points with jitter and shuffling applied, and the corresponding label.
"""

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.005, 0.005, dtype=tf.float64)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_points, test_labels))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.shuffle(len(test_points)).batch(BATCH_SIZE)

"""After this step, we have the following results :

-Each 3D object is represented by a fixed number of points.

-Keeping track of object classes.

-Adding noise and mixing points.

-Splitting the data into training and testing sets.

#**1.2 Model Architecture Implementation**

---


"""

#Application of a 1D convolution followed by batch normalization and a ReLU activation.
def conv_bn(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

#Application of a dense (fully connected) layer followed by batch normalization and a ReLU activation.
def dense_bn(x, filters):
    x = layers.Dense(filters)(x)
    x = layers.BatchNormalization(momentum=0.0)(x)
    return layers.Activation("relu")(x)

"""    The main idea here is to set a regularizer that enforces orthogonality on the weights of a layer by making
    weight matrices more orthogonal by adding a penalty to the loss function based on how far the weight matrices
    deviate from being orthogonal.
"""

class OrthogonalRegularizer(keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))

"""Applying a T-Net (Transformation Network) to learn an affine transformation for the input features.

    Parameters:
    - inputs: Input tensor with shape (batch_size, num_points, num_features).
    - num_features: Number of features (or dimensions) for the transformation matrix.

    Returns:
    - Tensor after applying the learned affine transformation to the input features.
"""

def tnet(inputs, num_features):

    # Initalise bias as the indentity matrix
    bias = keras.initializers.Constant(np.eye(num_features).flatten())
    reg = OrthogonalRegularizer(num_features)

    x = conv_bn(inputs, 32)
    x = conv_bn(x, 64)
    x = conv_bn(x, 512)
    x = layers.GlobalMaxPooling1D()(x)
    x = dense_bn(x, 256)
    x = dense_bn(x, 128)
    x = layers.Dense(
        num_features * num_features,
        kernel_initializer="zeros",
        bias_initializer=bias,
        activity_regularizer=reg,
    )(x)
    feat_T = layers.Reshape((num_features, num_features))(x)
    # Apply affine transformation to input features
    return layers.Dot(axes=(2, 1))([inputs, feat_T])

inputs = keras.Input(shape=(NUM_POINTS, 3))

x = tnet(inputs, 3)
x = conv_bn(x, 32)
x = conv_bn(x, 32)
x = tnet(x, 32)
x = conv_bn(x, 32)
x = conv_bn(x, 64)
x = conv_bn(x, 512)
x = layers.GlobalMaxPooling1D()(x)
x = dense_bn(x, 256)
x = layers.Dropout(0.3)(x)
x = dense_bn(x, 128)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="pointnet")
model.summary()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=["sparse_categorical_accuracy"],
)

model.fit(train_dataset, epochs=20, validation_data=test_dataset)

