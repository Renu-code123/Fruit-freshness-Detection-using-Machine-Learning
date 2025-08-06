from src.data_loader import get_data_generators
from src.model_builder import build_model
from utils.visualize import plot_training_history
import os

def train_model():
    train_dir = "data/fruit_dataset/train"
    train_gen, val_gen = get_data_generators(train_dir)
    model = build_model(num_classes=train_gen.num_classes)

    history = model.fit(train_gen, validation_data=val_gen, epochs=10)

    plot_training_history(history)

    os.makedirs("saved_models", exist_ok=True)
    model.save("saved_models/fruit_freshness_model.h5")
    print("Model saved!")
    return model, train_gen


