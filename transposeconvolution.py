import warnings
warnings.simplefilter('ignore')
import tensorflow as tf 

from keras.models import Model 

from keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D 

import numpy as np 

import matplotlib.pyplot as plt 

input_layer = Input(shape=(28, 28, 1))
conv_layer = Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same')(input_layer) #  creates a convolution layer with 32 filters.
transpose_conv_layer = Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid', padding='same')(conv_layer) # creates a transpose convolution layer for reconstructing the image.

#create the model by specifying the input and output layers.
model = Model(inputs=input_layer, outputs=transpose_conv_layer)
# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model 
X_train = np.random.rand(1000, 28, 28, 1) 
y_train = X_train # For reconstruction, the target is the input 
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2) 

# Test the model
X_test = np.random.rand(200, 28, 28, 1) 
y_test = X_test 
loss = model.evaluate(X_test, y_test) 
print(f'Test loss: {loss}')

# visualize the result
y_pred = model.predict(X_test) 
# Plot some sample images 
n = 10 # Number of samples to display 
plt.figure(figsize=(20, 4))
for i in range(n): 
    # Display original 
    ax = plt.subplot(2, n, i + 1) 
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.title("Original") 
    plt.axis('off') 
    # Display reconstruction 
    ax = plt.subplot(2, n, i + 1 + n) 
    plt.imshow(y_pred[i].reshape(28, 28), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
plt.show() 
