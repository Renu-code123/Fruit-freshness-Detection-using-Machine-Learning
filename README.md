# Fruit-freshness-Detection-using-Machine-Learning
This project presents a deep learning-based solution to classify fruits as fresh or rotten using Convolutional Neural Networks (CNNs). The aim is to build an intelligent system that can identify the freshness of fruits based on image inputs.
<hr>

📌 Project Overview:

1. Built a CNN model using TensorFlow and Keras for binary classification.

2 . Trained the model on a dataset containing images of various fruits categorized as fresh or rotten.

3  . Used Roboflow for image preprocessing, augmentation, and dataset pipeline generation.
<hr>

🧠 Tech Stack
1. Python
2.TensorFlow & Keras
3.NumPy
4. Matplotlib
5. VS Code (Development Environment)
6. Streamlit (for web demo deployment)
<hr>

📂 Dataset
Collected and prepared using Roboflow.

Includes labeled images for:

Fresh Apple  / Rotten Apple

Fresh Banana  / Rotten Banana 

Fresh Orange  / Rotten Orange 
<hr>

CNN tasks:
1. Upload raw images
2. Apply augmentation techniques
3 . Export dataset in TensorFlow-compatible format
<hr>

🧪 Model Architecture

Input Layer: Resized fruit images to 224x224

Convolutional Layers:

3 convolutional layers with MaxPooling

Flatten Layer

Dense Layer with Dropout (for regularization)

Output Layer: Softmax activation for multi-class classification
<hr>

Model Parameters:

Loss Function: categorical_crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy
<hr>

✅ Results:

Achieved >90% accuracy on the validation dataset.

Accurately classifies images into fresh or rotten categories across all fruit types.

