import tensorflow as tf 
from keras.models import Sequential 
from keras.layers import Dense, Flatten, Input
from keras.callbacks import Callback
from keras.models import Model
import numpy as np
# Step 1: Set Up the Environment
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize the pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0 

# Create a batched dataset for training
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
# Step 2: Define the Model
model = Sequential([
    Flatten(input_shape=(28, 28)),  # Flatten the input to a 1D vector
    Dense(128, activation='relu'),  # First hidden layer with 128 neurons and ReLU activation
    Dense(10)  # Output layer with 10 neurons for the 10 classes (digits 0-9)
])
# Step 3: Define Loss Function, Optimizer, and Metric
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # Loss function for multi-class classification
optimizer = tf.keras.optimizers.Adam()  # Adam optimizer for efficient training
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy()  # Metric to track accuracy during training
# Step 4: Implement the Custom Callback 
class CustomCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f'End of epoch {epoch + 1}, loss: {logs.get("loss")}, accuracy: {logs.get("accuracy")}')
# Step 5: Implement the Custom Training Loop with Custom Callback

epochs = 2
custom_callback = CustomCallback()  # Initialize the custom callback

for epoch in range(epochs):
    print(f'Start of epoch {epoch + 1}')
    
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        with tf.GradientTape() as tape:
            # Forward pass: Compute predictions
            logits = model(x_batch_train, training=True)
            # Compute loss
            loss_value = loss_fn(y_batch_train, logits)
        
        # Compute gradients
        grads = tape.gradient(loss_value, model.trainable_weights)
        # Apply gradients to update model weights
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        
        # Update the accuracy metric
        accuracy_metric.update_state(y_batch_train, logits)

        # Log the loss and accuracy every 200 steps
        if step % 200 == 0:
            print(f'Epoch {epoch + 1} Step {step}: Loss = {loss_value.numpy()} Accuracy = {accuracy_metric.result().numpy()}')
    
    # Call the custom callback at the end of each epoch
    custom_callback.on_epoch_end(epoch, logs={'loss': loss_value.numpy(), 'accuracy': accuracy_metric.result().numpy()})
    
    # Reset the metric at the end of each epoch
    accuracy_metric.reset_state()  # Use reset_state() instead of reset_states()
# Define the input layer
input_layer = Input(shape=(28, 28))  # Input layer with shape (28, 28)

# Define hidden layers
hidden_layer1 = Dense(64, activation='relu')(input_layer)  # First hidden layer with 64 neurons and ReLU activation
hidden_layer2 = Dense(64, activation='relu')(hidden_layer1)  # Second hidden layer with 64 neurons and ReLU activation
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)
model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 1: Redefine the Model for 20 features
model = Sequential([
    Input(shape=(20,)),  # Adjust input shape to (20,)
    Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    Dense(1, activation='sigmoid')  # Output layer for binary classification with sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 2: Generate Example Data
X_train = np.random.rand(1000, 20)  # 1000 samples, 20 features each
y_train = np.random.randint(2, size=(1000, 1))  # 1000 binary labels (0 or 1)

# Step 3: Train the Model
model.fit(X_train, y_train, epochs=10, batch_size=32)
# Example test data (in practice, use real dataset)
X_test = np.random.rand(200, 20)  # 200 samples, 20 features each
y_test = np.random.randint(2, size=(200, 1))  # 200 binary labels (0 or 1)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)

# Print test loss and accuracy
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')
