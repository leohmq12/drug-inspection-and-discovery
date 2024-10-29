import tensorflow as tf
from tensorflow.keras.models import load_model

#Load model
model = load_model(r'D:\drug-inspection-project\trained_model.h5')

#Load the model summary
model.summary()
