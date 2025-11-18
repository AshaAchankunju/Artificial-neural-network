import numpy as np
import matplotlib.pyplot as plt
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten
import tensorflow as tf


import os
from PIL import Image
import numpy as np
# Load the VGG16 model pre-trained on ImageNet
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Create a new model and add the base model and new layers
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')  # Change to the number of classes you have
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Create directories if they don't exist
os.makedirs('sample_data/class_a', exist_ok=True)
os.makedirs('sample_data/class_b', exist_ok=True)

# Create 10 sample images for each class
for i in range(10):
    # Create a blank white image for class_a
    img = Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255)
    img.save(f'sample_data/class_a/img_{i}.jpg')

    # Create a blank black image for class_b
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    img.save(f'sample_data/class_b/img_{i}.jpg')

print("Sample images created in 'sample_data/'")

# Load and preprocess the dataset

train_datagen = tf.keras.utils.image_dataset_from_directory(
    'sample_data',
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)


# Verify if the generator has loaded images correctly
# Optional: Normalize pixel values to [0,1]
train_datagen = train_datagen.map(lambda x, y: (x/255.0, y))

# Optional: Prefetch for performance
train_datagen = train_datagen.prefetch(tf.data.AUTOTUNE)

# Train the model directly
model.fit(train_datagen, epochs=10)


# Unfreeze the top layers of the base model 
for layer in base_model.layers[-4:]:
    layer.trainable = True 

# Compile the model again 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

# Train the model again 
model.fit(train_datagen, epochs=10) 