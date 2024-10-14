# Fatigue_Detection_CNN
This model has demonstrated superior performance, achieving an 
impressive accuracy of 99.17% on the DDD dataset, showcasing 
its effectiveness in detecting drowsiness in images. Additionally, 
the model has shown a high rate of correct predictions when tested 
on new, unseen data, further proving its potential for accurate 
drowsiness detection.

# Model Architecture
![CNN-Model](https://github.com/user-attachments/assets/5227a5f3-1dab-4f2c-9b09-a62fb0e3e4ba)

It is designed for binary classification, begins with an input layer accepting grayscale images of size 64x64 pixels.
Then, it has two consecutive blocks of two convolutional layers with 32 and 64 filters respectively, each followed by
a batch normalization layer for stabilizing training and a max-pooling layer to reduce dimensionality. Additionally, 
dropout layers with a rate of 0.3 are interspersed to mitigate overfitting. The feature maps are then flattened and 
passed through several dense layers with decreasing units (256, 128, and 84), each followed by batch normalization, 
except for the layer with 128 units. Finally, the output layer consists of a single neuron with a sigmoid activation function.

# Model Tested
![Tested7](https://github.com/user-attachments/assets/71ceefbb-6025-4295-8cab-8f530503a765)
![Tested8](https://github.com/user-attachments/assets/2ad8046e-9d4b-4461-8ad1-1c008d37b161)
