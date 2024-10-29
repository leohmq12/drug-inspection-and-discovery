import os
import h5py
from PIL import Image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Define the folder paths
image_folder = r'D:\drug-inspection-project\data\vencerlanz09\pharmaceutical-drugs-and-vitamins-dataset-v2\versions\1\Capsure Dataset\Train Image'
h5_file = r'D:\drug-inspection-project\dataset-training.h5'

# Empty list to store image data and labels
image_data = []
labels = []  # You will need labels if this is a classification task

# Loop through the folders and read images
for root, dirs, files in os.walk(image_folder):
    for file in files:
        # Check if the file is an image (you can add more formats if needed)
        if file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            img = img.resize((128, 128))  # Resize all images to the same size
            img_array = np.array(img)
            image_data.append(img_array)

            # For simplicity, use folder name as a label (assuming folder name corresponds to class)
            label = os.path.basename(root)
            labels.append(label)

# Convert the list of images and labels to numpy arrays
image_data = np.array(image_data)

# Convert labels to categorical format (assuming this is for classification)
unique_labels = list(set(labels))
label_indices = [unique_labels.index(label) for label in labels]
labels_array = to_categorical(label_indices, num_classes=len(unique_labels))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(image_data, labels_array, test_size=0.2, random_state=42)

# Build a simple CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(unique_labels), activation='softmax')  # Number of output neurons equal to number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Save the model (architecture + weights)
model.save('D:\\drug-inspection-project\\trained_model.h5')

print(f"Model successfully saved to 'D:\\drug-inspection-project\\trained_model.h5'")
