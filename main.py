from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict_custom_image

if __name__ == "__main__":
    model, train_generator = train_model()
    evaluate_model(model)
    predict_custom_image(model)


