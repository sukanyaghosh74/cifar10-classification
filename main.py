import os
import argparse
from src.train import train_model
from src.transfer_learning import train_transfer_model

def main(model_type):
    if model_type == "cnn":
        print("Training CNN model...")
        train_model()
    elif model_type == "transfer":
        print("Training Transfer Learning model with VGG16...")
        train_transfer_model()
    else:
        print("Invalid model type. Use 'cnn' or 'transfer'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-10 Image Classification")
    parser.add_argument("model", type=str, help="Model type: 'cnn' or 'transfer'")
    args = parser.parse_args()
    main(args.model)