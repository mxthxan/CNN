import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping


np.random.seed(42)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


train_path = (r"D:/mamo/train")
valid_path = (r"D:/mamo/valid")
test_path = (r"D:/mamo/test")


train_gen = ImageDataGenerator(rescale=1.0 / 255, 
                               rotation_range=20, 
                               width_shift_range=0.2, 
                               height_shift_range=0.2, 
                               horizontal_flip=True)

valid_gen = ImageDataGenerator(rescale=1.0 / 255)
test_gen = ImageDataGenerator(rescale=1.0 / 255)


train_data = train_gen.flow_from_directory(
    train_path, 
    target_size=IMAGE_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary'
)

valid_data = valid_gen.flow_from_directory(
    valid_path, 
    target_size=IMAGE_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary'
)

test_data = test_gen.flow_from_directory(
    test_path, 
    target_size=IMAGE_SIZE, 
    batch_size=BATCH_SIZE, 
    class_mode='binary'
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Sigmoid for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Set early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=20,
    steps_per_epoch=len(train_data),
    validation_steps=len(valid_data),
    callbacks=[early_stop]
)

# Plot accuracy and loss
def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

plot_history(history)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_data)
print(f"Test Accuracy: {test_acc:.2f}")

# Confusion matrix and classification report
test_data.reset()
predictions = model.predict(test_data, steps=len(test_data), verbose=1)
y_pred = np.where(predictions > 0.5, 1, 0)

print("Confusion Matrix:")
print(confusion_matrix(test_data.classes, y_pred))

print("Classification Report:")
print(classification_report(test_data.classes, y_pred, target_names=['Benign', 'Malignant']))

# Save the model
model.save('mammogram_cancer_model.h5')

# Function to predict new images
def predict_image(image_path):
    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    if prediction[0][0] > 0.5:
        print("Prediction: Malignant")
    else:
        print("Prediction: Benign")

# Test with a new image
predict_image(r"D:/mamo/test/1/729_748167281_png.rf.496a8581ecdbc0dfd0ffbb1affc2a1d1.jpg")
