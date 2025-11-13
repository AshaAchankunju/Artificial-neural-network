import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load a sample image
img_path = 'sample.jpg'
img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
x = tf.keras.utils.img_to_array(img)
x = tf.expand_dims(x, 0)  # Add batch dimension

# Define a data augmentation pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomTranslation(0.2, 0.2),
])

# Apply augmentation and visualize
augmented_images = data_augmentation(x)

plt.figure(figsize=(10, 5))
for i in range(4):
    plt.subplot(1, 4, i+1)
    augmented = data_augmentation(x)
    plt.imshow(tf.cast(augmented[0], tf.uint8))
    plt.axis('off')
plt.show()

# Load multiple images
image_paths = [
    'sample_images/training_images1.jpg',
    'sample_images/training_images2.jpg',
    'sample_images/training_images3.jpg'
]

training_images = []
for path in image_paths:
    img = tf.keras.utils.load_img(path, target_size=(224, 224))  # <-- correct
    img_array = tf.keras.utils.img_to_array(img)
    training_images.append(img_array)

training_images = tf.convert_to_tensor(training_images)

# Normalize images
training_images = tf.image.per_image_standardization(training_images)


# Normalize images (per image)
training_images = tf.image.per_image_standardization(training_images)

# Add random noise function
def add_random_noise(image):
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=10.0)
    image = image + noise
    image = tf.clip_by_value(image, 0, 255)
    return image

# Apply noise to images
noisy_images = tf.map_fn(add_random_noise, training_images)

# Visualize noisy images
plt.figure(figsize=(10, 5))
for i in range(len(noisy_images)):
    plt.subplot(1, len(noisy_images), i+1)
    plt.imshow(tf.cast(noisy_images[i], tf.uint8))
    plt.axis('off')
plt.show()
