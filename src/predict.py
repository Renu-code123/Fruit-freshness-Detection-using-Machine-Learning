from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

# ✅ Add your class names here in correct order
class_names = [
    'Fresh Apple',     # class 0
    'Fresh Banana',    # class 1
    'Fresh Orange',    # class 2
    'Rotten Apple',    # class 3
    'Rotten Banana',   # class 4
    'Rotten Orange'    # class 5
]

def predict_custom_image(model):
    print("\n📂 Please select an image file to predict...")

    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])

    if not file_path:
        print("❌ No file selected.")
        return

    try:
        img = image.load_img(file_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # ✅ Directly use class_names to get the label
        predicted_label = class_names[class_idx]

        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Prediction: {predicted_label}\nConfidence: {confidence:.2f}%")
        plt.show()

    except Exception as e:
        print(f"❌ Error processing image: {e}")
