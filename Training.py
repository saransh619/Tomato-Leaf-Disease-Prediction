from tensorflow.compat.v1 import ConfigProto, InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

tf.compat.v1.disable_eager_execution()

import matplotlib.pyplot as plt

#basic cnn
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation = 'relu'))
classifier.add(Dropout(0.5))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units = 10, activation = 'softmax')) 

#Model Summary
classifier.summary()

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale=1./255) 
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./Dataset/train/', # relative path from working directoy
                                                 target_size = (128, 128),
                                                 batch_size = 6, class_mode = 'categorical')

valid_set = test_datagen.flow_from_directory('./Dataset/val/', # relative path from working directoy
                                             target_size = (128, 128), 
                                        batch_size = 3, class_mode = 'categorical')

test_set = test_datagen.flow_from_directory(
        './Dataset/test/',
        target_size = (128, 128),
        class_mode = 'categorical',
        color_mode = "rgb"
)

labels = (training_set.class_indices)
print(labels)

history = classifier.fit(training_set,
                         epochs = 35,
                         validation_data=valid_set
                         )


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# plot the train and val curve for accuracy
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(35), acc, label='Training Accuracy')
plt.plot(range(35), val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.subplot(1, 2, 2)
plt.plot(range(35), loss, label='Training Loss')
plt.plot(range(35), val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

valid_set.reset()

# Evaluate on Validation data
#Calculate Model Accuracy
print("[INFO] Calculating model accuracy")
scores = classifier.evaluate(valid_set)
print(f"Test Loss: {scores[0]}")
print(f"Test Accuracy: {scores[1]*35}")
# scores = classifier.evaluate(valid_set)
# print("%s%s: %.2f%%" % ("evaluate ",classifier.metrics_names[1], scores[1]*35))


#saved model using h5 
classifier_json=classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)
# serialize weights to HDF5
    classifier.save_weights("my_model_weights.h5")
    classifier.save("model.h5")
    print("Saved model to disk")

