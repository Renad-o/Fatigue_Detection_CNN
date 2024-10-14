from keras.models import load_model
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load CNN model
model = load_model('DrowsinessDetection.h5')

# Define classes (order is important for label consistency)
class_names = ['Drowsy', 'Not Drowsy']

# Image size should match your model's input shape
image_size = (64, 64)


# Prediction on test images
for i, image_path in enumerate(tqdm(glob('/Users/Renad/Desktop/GP-Models/ToTest/*'))):  # Iterate over test images
       img = Image.open(image_path).convert('L').resize(image_size)  # Load and resize as grayscale
       img_array = np.array(img) / 255.0
       img_array = np.expand_dims(img_array, axis=-1) # Add channel dimension (for grayscale)
       img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

       # Make prediction
       prediction = model.predict(img_array)[0] # Get the prediction for the current image

       # Determine the predicted class
       predicted_class = 1 if prediction > 0.4 else 0
       class_name = class_names[predicted_class]

       # Display the image and prediction
       plt.imshow(img_array[0, :, :, 0])
       plt.title(f"Predicted: {class_name} (Probability: {prediction[0]:.2f})")
       plt.axis('off') #Turn off axis labels
       plt.savefig('Tested' + str(i) + '.png' )
       plt.show()


