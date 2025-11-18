import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.applications import VGG16
from keras.models import Sequential, clone_model
from keras.layers import Flatten, Dense

# -----------------------------
# 1. Create sample dataset
# -----------------------------
os.makedirs('sample_data/class_a', exist_ok=True)
os.makedirs('sample_data/class_b', exist_ok=True)

for i in range(10):
    Image.fromarray(np.ones((224, 224, 3), dtype=np.uint8) * 255).save(f'sample_data/class_a/img_{i}.jpg')
    Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)).save(f'sample_data/class_b/img_{i}.jpg')

print("Sample images created.")

# -----------------------------
# 2. Load dataset with validation split
# -----------------------------
batch_size = 4
img_size = (224, 224)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    'sample_data',
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    'sample_data',
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# Normalize pixel values
train_dataset = train_dataset.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.map(lambda x, y: (x / 255.0, y)).prefetch(tf.data.AUTOTUNE)

# -----------------------------
# 3. Build model
# -----------------------------
base_model = VGG16(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False  # freeze all layers

model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------
# 4. Train model
# -----------------------------
history = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# -----------------------------
# 5. Fine-tune last 4 layers of VGG16
# -----------------------------
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history_finetune = model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# -----------------------------
# 6. Experiment with different optimizers
# -----------------------------
def train_with_optimizer(model, optimizer, train_ds, val_ds, epochs=5):
    model_copy = clone_model(model)
    model_copy.set_weights(model.get_weights())
    model_copy.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model_copy.fit(train_ds, validation_data=val_ds, epochs=epochs)

# SGD
history_sgd = train_with_optimizer(model, 'sgd', train_dataset, val_dataset)

# RMSprop
history_rmsprop = train_with_optimizer(model, 'rmsprop', train_dataset, val_dataset)

# -----------------------------
# 7. Plot results
# -----------------------------
plt.plot(history.history['loss'], label='Training Loss Adam')
plt.plot(history.history['val_loss'], label='Validation Loss Adam')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history_sgd.history['accuracy'], label='Training Accuracy SGD')
plt.plot(history_sgd.history['val_accuracy'], label='Validation Accuracy SGD')
plt.title('Training and Validation Accuracy with SGD')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(history_rmsprop.history['accuracy'], label='Training Accuracy RMSprop')
plt.plot(history_rmsprop.history['val_accuracy'], label='Validation Accuracy RMSprop')
plt.title('Training and Validation Accuracy with RMSprop')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -----------------------------
# 8. Evaluate on the same dataset (tiny demo)
# -----------------------------
test_loss, test_accuracy = model.evaluate(val_dataset)
print(f'Test Accuracy: {test_accuracy*100:.2f}%')
print(f'Test Loss: {test_loss:.4f}')
