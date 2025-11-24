import sys

# Increase recursion limit to prevent potential issues
sys.setrecursionlimit(100000)
# Step 2: Import necessary libraries
import keras_tuner as kt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.datasets import mnist
from keras.optimizers import Adam
import os
import warnings
from tensorflow import keras
from sklearn.model_selection import train_test_split
# Suppress all Python warnings
warnings.filterwarnings('ignore')

# Set TensorFlow log level to suppress warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = all logs, 1 = filter out INFO, 2 = filter out INFO and WARNING, 3 = ERROR only
# Load the MNIST dataset
(x_all, y_all), _ = keras.datasets.mnist.load_data()

# Flatten and normalize the images
x_all = x_all.reshape((x_all.shape[0], -1)).astype("float32") / 255.0

# Split into train+val and test (80/20)
x_temp, x_test, y_temp, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

# Split train+val into train and validation (75/25 of 80% = 60/20 overall)
x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)

# Step 3: Load and preprocess the MNIST dataset
(x_train, y_train), (x_val, y_val) = mnist.load_data()
x_train, x_val = x_train / 255.0, x_val / 255.0
# mnist.load_data(): Loads the dataset, returning training and validation splits.
# x_train / 255.0: Normalizes the pixel values to be between 0 and 1.
# print(f'...'): Displays the shapes of the training and validation datasets.
print(f'Training data shape: {x_train.shape}')
print(f'Validation data shape: {x_val.shape}')

# Define a model-building function 
# Create a function build_model that takes a HyperParameters object as input.
# Use the HyperParameters object to define the number of units in a dense layer and the learning rate for the optimizer.
# Compile the model with sparse categorical cross-entropy loss and Adam optimizer.
def build_model(hp):
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(units=hp.Int('units', min_value=32, max_value=512, step=32), activation='relu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Create a RandomSearch Tuner 
# Configuring the hyperparameter search¶
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=2,
    directory='my_dir',
    project_name='intro_to_kt'
)

# Display a summary of the search space 
# epochs=5: Each trial is trained for 5 epochs.
# validation_data=(x_val, y_val): The validation data to evaluate the model's performance during the search.
tuner.search_space_summary()

# Run the hyperparameter search 
tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val)) 

# Display a summary of the results 
tuner.results_summary() 





# Analyzing and using the best hyperparameters
# Step 1: Retrieve the best hyperparameters 
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0] 
print(f""" 

The optimal number of units in the first dense layer is {best_hps.get('units')}. 

The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}. 

""") 

# Step 2: Build and Train the Model with Best Hyperparameters 
model = tuner.hypermodel.build(best_hps) 
model.fit(x_train, y_train, epochs=5, validation_split=0.2) 

# Evaluate the model on the test set 
test_loss, test_acc = model.evaluate(x_val, y_val) 
print(f'Test accuracy: {test_acc}') 

