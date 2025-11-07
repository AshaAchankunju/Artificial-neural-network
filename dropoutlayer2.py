from keras.layers import Dropout, Input, Dense
from keras.models import Model
import numpy as np
input_layer=Input(shape=(20,))
hidden_layer1=Dense(64, activation='relu')(input_layer)
dropout1=Dropout(0.5)(hidden_layer1)
hidden_layer2=Dense(64,activation='relu')(dropout1)
dropout2=Dropout(0.5)(hidden_layer2)
output_layer=Dense(1,activation='sigmoid')(dropout2)
model=Model(inputs=input_layer,outputs=output_layer)
model.summary()
model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
X_train = np.random.rand(1000, 20) 
y_train = np.random.randint(2, size=(1000, 1)) 
X_test = np.random.rand(200, 20) 
y_test = np.random.randint(2, size=(200, 1)) 
model.fit(X_train, y_train,epochs=10 , batch_size=32)

loss,accuracy=model.evaluate(X_test, y_test)
print(f'Test loss: {loss}') 
print(f'Test accuracy:{accuracy}')