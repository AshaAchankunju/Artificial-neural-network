from keras.layers import Dropout, Dense, Input
from keras.models import Model

# Define the input layer
input_layer = Input(shape=(20,))

# Add a hidden layer
hidden_layer = Dense(64, activation='relu')(input_layer)

# Add a Dropout layer
dropout_layer = Dropout(rate=0.5)(hidden_layer)

# Add another hidden layer after Dropout
hidden_layer2 = Dense(64, activation='relu')(dropout_layer)

# Define the output layer
output_layer = Dense(1, activation='sigmoid')(hidden_layer2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Summary of the model
model.summary()