# Fruit-freshness-Detection-using-Machine-Learning
This project is a deep learning-based solution to classify fruits as fresh or rotten using Convolutional Neural Networks (CNNs). The goal is to provide an intelligent system that can automatically determine the freshness of fruits from their images.

📌 Project Overview
The model is trained using a dataset containing images of various fruits in two categories: fresh and rotten. The CNN model is developed using TensorFlow and Keras to perform binary classification.

We used Roboflow to preprocess, augment, and generate a dataset pipeline that makes model training efficient and robust.

🧠 Tech Stack
Python 🐍

TensorFlow & Keras 🧠

NumPy & Matplotlib 📊

VS Code for deployment(IDE)

Streamlit (for demo app – optional)

📂 Dataset
The dataset used for training and validation was obtained from Roboflow and contains labeled images in the categories:

Fresh Apple, Rotten Apple

Fresh Banana, Rotten Banana

Fresh Orange, Rotten Orange

Roboflow helped with:

Uploading raw images

Image augmentation

Exporting dataset in TensorFlow-ready format

🧪 Model Architecture
Input Layer: Image (resized to 224x224)

3 Convolutional layers with MaxPooling

Flatten layer

Dense layer with Dropout

Output layer with Softmax (for multi-class classification)

Loss Function: categorical_crossentropy
Optimizer: Adam
Metric: Accuracy

✅ Results
The model achieved over 90% accuracy on the validation set. It successfully classifies fruit images into fresh or rotten categories.
