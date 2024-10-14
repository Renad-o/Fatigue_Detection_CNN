import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
import seaborn as sns


X = []
Y = []

# Load and preprocess images (with tqdm for progress bar)
for i in tqdm(glob('/Processed_eye/Non_Drowsy/*')):
    temp = np.array(Image.open(i).convert('L').resize((64, 64)))
    X.append(temp)
    Y.append(1)

for i in tqdm(glob('/Processed_eye/Drowsy/*')):
    temp = np.array(Image.open(i).convert('L').resize((64, 64)))
    X.append(temp)
    Y.append(0)

# Convert lists to NumPy arrays and normalize X
X = np.array(X)
X = X / 255.0  # Normalize pixel values to [0, 1]
Y = np.array(Y)

print("X shape:", X.shape)  # Print the shape of X

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Conv2D, BatchNormalization, MaxPooling2D,Dropout, Flatten

model = Sequential([
      Input(shape=(64, 64, 1)),

      Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'relu'),
      Conv2D(filters = 32, kernel_size = 5, strides = 1, activation = 'relu', use_bias=False),
      BatchNormalization(),
      MaxPooling2D(strides = 2),
      Dropout(0.3),

      Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu'),
      Conv2D(filters = 64, kernel_size = 3, strides = 1, activation = 'relu', use_bias=False),
      BatchNormalization(),
      MaxPooling2D(strides = 2),
      Dropout(0.3),

      Flatten(),
      Dense(units  = 256, activation = 'relu', use_bias=False),
      BatchNormalization(),

      Dense(units = 128, use_bias=False, activation = 'relu'),

      Dense(units = 84, use_bias=False, activation = 'relu'),
      BatchNormalization(),
      Dropout(0.3),

      Dense( 1, activation = 'sigmoid')
  ])

model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])

callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='DrowsinessDetectionFace.h5',
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
    verbose =1)
history = model.fit(x_train, y_train, validation_split=0.2, epochs=30, batch_size=16, callbacks=callback)

# Model Evaluation
loss, accuracy = model.evaluate(x_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)

h = history
#-------------------Plot loss value------------------
plt.plot(h.history['loss'], label='train loss')
plt.plot(h.history['val_loss'], label='validation loss')
plt.legend()
plt.savefig('LossValue.png')
plt.show()

#----------------Plot accuracy value---------------
plt.plot(h.history['accuracy'], label='train accuracy')
plt.plot(h.history['val_accuracy'], label='validation accuracy')
plt.legend()
plt.savefig('AccuracyValue.png')
plt.show()

# Predictions and Probabilities
y_pred_probs = model.predict(x_test)
y_pred_classes = (y_pred_probs > 0.5).astype(int)

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred_classes, target_names=['Drowsy', 'Non_Drowsy']))

#----------------Confusion Matrix----------------
cm = confusion_matrix(y_test, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Drowsy', 'Non_Drowsy'], yticklabels=['Drowsy', 'Non_Drowsy'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('ConfusionMatrix.png')
plt.show()


#----------------ROC-------------------
#calculate fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)

#create function for plotting ROC curve
def plot_ROC_curve(fpr, tpr):
    plt.plot(fpr, tpr, color="orange", label='ROC')
    plt.plot([0,1], [0,1], color='darkblue', linestyle='--', label='Guessing')
    #customize
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig('CNN_ROC_Curve.png')
    plt.show()

plot_ROC_curve(fpr, tpr)

print('The AUC score is: ', roc_auc_score(y_test, y_pred_probs))
