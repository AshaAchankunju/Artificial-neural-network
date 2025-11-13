import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import os

# --- Step 0: Create a sample image ---
os.makedirs("dataset/samples", exist_ok=True)
image = Image.new('RGB', (224, 224), color=(255, 255, 255))
draw = ImageDraw.Draw(image)
draw.rectangle([(50, 50), (174, 174)], fill=(255, 0, 0))
image.save('dataset/samples/sample1.jpg')

# --- Step 1: Load dataset ---
from keras.utils import image_dataset_from_directory

dataset = image_dataset_from_directory(
    'dataset',
    labels=None,           # None if unsupervised / no classes
    image_size=(224, 224),
    batch_size=1,
    shuffle=False
)

# --- Step 2: Define data augmentation layers ---
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),         # normalize images to [0,1]
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1)
])

# --- Step 3: Apply augmentation and visualize ---
plt.figure(figsize=(8, 8))
for i, img_batch in enumerate(dataset.take(4)):
    augmented_images = data_augmentation(img_batch)
    plt.subplot(2, 2, i+1)
    plt.imshow(tf.squeeze(augmented_images[0]))  # remove batch dim
    plt.axis('off')
plt.show()

# --- Step 4: Optional - add custom noise ---
def add_random_noise(image):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1)
    return tf.clip_by_value(image + noise, 0.0, 1.0)

dataset_noisy = dataset.map(lambda x: add_random_noise(tf.cast(x, tf.float32)/255.0))

# Visualize noisy images
plt.figure(figsize=(8, 8))
for i, img_batch in enumerate(dataset_noisy.take(4)):
    plt.subplot(2, 2, i+1)
    plt.imshow(tf.squeeze(img_batch[0]))
    plt.axis('off')
plt.show()
