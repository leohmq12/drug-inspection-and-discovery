import os
import h5py
from PIL import Image
import numpy as np

#Define the folder paths
image_folder = 'D:\drug-inspection-project\data\vencerlanz09\pharmaceutical-drugs-and-vitamins-dataset-v2\versions\1\Capsure Dataset\Train Image'
h5_file = 'D:\drug-inspection-project\dataset-training.h5'

#Empty list to store image data
image_data = []

for root, dirs, files in os.walk(image_folder):
    for file in files:
        # Check if the file is an image (you can add more formats if needed)
        if file.endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff')):
            img_path = os.path.join(root, file)
            img = Image.open(img_path)
            img = img.resize((128, 128))  # Resize all images to the same size, adjust as needed
            img_array = np.array(img)
            image_data.append(img_array)

#Conversion to numpy array
image_data = np.array(image_data)

#Create h5 file with the images stored in numpy array
with h5py.File(h5_file, 'w') as hf:
    hf.create_dataset('images',data=image_data)

print(f"Images successfully saved to {h5_file}")
