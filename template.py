import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.cross_validation import train_test_split
import numpy as np

image_data = np.load('images.npy')
label_data = np.load('labels.npy')


x = image_data.reshape([6500,784])
y = keras.utils.to_categorical(label_data, num_classes=10)

#Splitting Data into train, val, and test

x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.40)
x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.625)


# Model Template

model = Sequential() # declare model

model.add(Dense(25, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
model.add(Dense(23, kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy', 
              metrics=['accuracy'])



# Train Model
history = model.fit(x_train, y_train,
                    validation_data = (x_val, y_val), 
                    epochs=100, 
                    batch_size=512)


# Report Results

print(history.history)
scores = model.evaluate(x_val, y_val, batch_size=975)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
prediction = model.predict(x_test)

i = 0
while i < 16250:
    print("Predicted: ", prediction[i])
    print("Expected: ", y_test[i])
    i+=1
    



