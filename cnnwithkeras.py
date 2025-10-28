import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.utils import to_categorical

from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test)= mnist.load_data()
X_train=X_train.reshape(X_train.shape[0],28,28,1).astype('float32')
X_test=X_test.reshape(X_test.shape[0],28,28,1).astype('float32')
X_train= X_train/255
X_test=X_test/255
 
# converting to binary categories
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
num_classes=y_test.shape[1]

def convolutional_model():
    model=Sequential()
    model.add(Input(shape=(28,28,1)))
    #one set of  convolutional and pooling layer
    model.add(Conv2D(16,(5,5),strides=(1,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(100,activation='relu'))
    model.add(Dense(num_classes,activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model=convolutional_model()
model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=10, batch_size=200, verbose=2)

scores=model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: {} \n Error: {}".format(scores[1],100-scores[1]*100))