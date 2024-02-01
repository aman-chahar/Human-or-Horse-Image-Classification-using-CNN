**Image Classification with Convolutional Neural Networks 🐴👨‍🦳**

Excited to share my latest deep learning project focused on binary image classification using Convolutional Neural Networks (CNN). The task involved distinguishing between images of horses and humans, leveraging TensorFlow datasets. 🚀

**Project Overview:**
For this assignment, I created a robust CNN model to perform binary classification on a dataset containing images of horses and humans. The dataset was conveniently accessed through TensorFlow datasets, providing a seamless environment for experimentation.

**Key Steps in the Project:**

1. **Data Preprocessing:**
   - Downloaded the necessary dependencies, including TensorFlow datasets.
   - Loaded and processed the 'horses_or_humans' dataset, organizing it into training and testing sets.

```python
# Downloading the Dependencies
!pip install tensorflow_datasets

# Importing required libraries
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import tensorflow_datasets as tfds
import os
...
# Data Visualization
nrows = 2
ncols = 6
fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4) # 16x8
...
plt.show()
```

2. **Building the CNN Model:**
   - Constructed a CNN model with three convolutional layers, each followed by max-pooling.
   - Added dense layers for flattening and fully connected networks.
   - Compiled the model using the Adamax optimizer and binary cross-entropy loss.

```python
# Building CNN Model
def get_model():
  model = Sequential()
  # 1st layer CNN
  model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu', input_shape=(300,300,3)))
  model.add(MaxPool2D(pool_size=2))
  ...
  model.compile(optimizer=adamax, loss='binary_crossentropy', metrics=['accuracy'])
  ...
```

3. **Training and Evaluation:**
   - Trained the model for five epochs, monitoring both training and validation accuracy and loss.
   - Visualized the training process using matplotlib.

```python
# Training and Evaluation
history = model.fit(train_generator, epochs=5, validation_data=test_generator)

# Plot loss and accuracy curves
fix, ax = plt.subplots(2,1)
...
```

4. **Model Saving and Loading:**
   - Saved the trained model as 'horse-or-human.h5'.
   - Demonstrated model loading and prediction on a sample image.

```python
# Save and load model
model.save('horse-or-human.h5')

from tensorflow.keras.models import load_model
model_load = load_model('horse-or-human.h5')

# Image Prediction
img = image.load_img('/content/horse-or-human/test/horses/horses_154.jpg', target_size=image_size)
...
prediction = model_load.predict(img)
print(prediction)
...
```

**Results and Conclusion:**
The CNN model demonstrated a remarkable accuracy of 74% in distinguishing between horses and humans. The project showcases the power of deep learning in image classification tasks, paving the way for exciting applications in various domains. 🌐💡

Stay tuned for more updates on my journey into the world of deep learning and AI! Feel free to connect and share your thoughts. 🤝 #DeepLearning #ImageClassification #CNN #TensorFlow #AI

Looking forward to your feedback! 😊
