import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score
from sklearn.cross_validation import train_test_split
from keras.models import load_model
import matplotlib.pyplot as plt
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
model.add(Activation('tanh'))
model.add(Dense(100, kernel_initializer='he_normal'))
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
scores = model.evaluate(x_val, y_val, batch_size=512)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
prediction = model.predict(x_test)
print("")
# Plot a Graph showing training set and validation accuracy over epoch
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='lower right')
plt.show()

count = 0
i = 0
j = 0
error = [0,0,0]
pTemp = [0,0,0,0,0,0,0,0,0,0]

# Get Wrongly Classified Image
while i < 1625:
    j = 0
    for p in prediction[i]:
        pTemp[j] = round(p)
        j+=1
    prediction[i] = pTemp
    a = 0
    tTemp = y_test[i]
    if count < 3:
        for predicted in prediction[i]:
            if tTemp[a] != predicted:
                error[count] = i
                count += 1
                break
            a += 1    
    i+=1

size = 0
confusion_matrix = [[0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0,0,0,0]]

while size < 1625:
    preMarker = 0
    testMarker = 0
    for test in y_test[size]:
        if test == 1:
            for pre in prediction[size]:
                if pre == 1:
                    temp = confusion_matrix[testMarker][preMarker]
                    confusion_matrix[testMarker][preMarker] = temp + 1
                if preMarker < 9:
                    preMarker += 1
        if testMarker < 9:
            testMarker += 1
    size += 1

print("Confusion Matrix")
print("")
print("The row is Prediction Label and the column is True Label")
print("")
print(np.matrix(confusion_matrix))


# Show the Image
for e in error:
    label = y_test
    pixels = x_test[e]
    pixels = np.array(pixels, dtype='uint8')
    pixels = pixels.reshape(28,28)
    plt.title(prediction[e])
    plt.imshow(pixels, cmap= 'gray')
    plt.show()
    
model.save('trained_model.h5')
del model

model = load_model('trained_model.h5')

