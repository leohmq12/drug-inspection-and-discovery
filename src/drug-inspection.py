import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model for prediction
loaded_model = load_model('D:\\drug-inspection-project\\trained_model.h5')

from tkinter import Tk, filedialog

# Function to load an image using a file dialog
def load_image():
    Tk().withdraw()  # Close the root window
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        raise FileNotFoundError("No image selected.")
    return file_path


#Load and preprocess the drug image for prediction
img_path = load_image()
img = cv2.imread(img_path)

# Check if the image was successfully loaded
if img is None:
    raise FileNotFoundError(f"Image not found at {img_path}. Check the file path.")

# Resize and preprocess the image
img = cv2.resize(img, (128, 128))

# Convert the image to a NumPy array and add a batch dimension
img = np.expand_dims(img, axis=0)

# Normalize the image
img = img / 255.0

# Prediction using the model
prediction = loaded_model.predict(img)

# Your code for interpreting the prediction (e.g., thresholding for classification)

predicted_class = np.argmax(prediction, axis=1)[0]
if predicted_class == 1:  # Adjust class labels based on your model's output
    print("Authentic Drug")
else:
    print("Counterfeit Drug")
