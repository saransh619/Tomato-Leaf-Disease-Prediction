import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

print("🚀 Initializing MobileNetV2 Transfer Learning...")

# 1. Image Data Generators
# We keep the 1./255 rescale to ensure 100% compatibility with app.py!
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True,
    rotation_range=20, # Add a little rotation for robustness
    fill_mode='nearest'
)

valid_datagen = ImageDataGenerator(rescale=1./255)

print("\nLoading Training Data...")
training_set = train_datagen.flow_from_directory(
    './Dataset/Training/',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical'
)

print("\nLoading Validation Data...")
valid_set = valid_datagen.flow_from_directory(
    './Dataset/Validation/',
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical'
)

# 2. Base Model (MobileNetV2)
print("\nDownloading Pre-trained MobileNetV2 Base...")
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(128, 128, 3)
)

# Freeze the base model so we don't destroy the pre-trained smarts!
base_model.trainable = False

# 3. Custom Classification Head
classifier = Sequential([
    base_model,
    GlobalAveragePooling2D(), # Far superior to Flatten() for CNNs
    Dropout(0.3), # Prevent overfitting
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(10, activation='softmax') # Our 10 tomato diseases!
])

classifier.summary()

# 4. Compile the Model
classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Callbacks (Smart Training)
# Stop training early if accuracy maxes out
early_stop = EarlyStopping(monitor='val_accuracy', patience=4, restore_best_weights=True)
# Always save the best performing model
checkpoint = ModelCheckpoint("model.h5", monitor='val_accuracy', save_best_only=True, verbose=1)

# 6. Train the Model!
print("\n🔥 Starting High-Performance Training...")
history = classifier.fit(
    training_set,
    epochs=20, # 20 is plenty for Transfer Learning
    validation_data=valid_set,
    callbacks=[early_stop, checkpoint]
)

# 7. Evaluate the final accuracy
print("\n[INFO] Calculating final model accuracy on Validation Set...")
scores = classifier.evaluate(valid_set)
print(f"✅ Final Test Accuracy: {scores[1]*100:.2f}%")

print("🎉 Training Complete! A highly accurate model.h5 has been saved.")
