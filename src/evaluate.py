from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def evaluate_model(model):
    test_dir = "data/fruit_dataset/test"
    test_gen = ImageDataGenerator(rescale=1./255)

    test_generator = test_gen.flow_from_directory(
        test_dir,
        target_size=(150, 150),
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    loss, acc = model.evaluate(test_generator)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
