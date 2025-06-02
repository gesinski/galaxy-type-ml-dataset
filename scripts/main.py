from models.cnn import CNNModel
from models.mlp import MLPModel
from models.resnet import get_resnet18_model
from scripts.utils import load_data
from scripts.train import train_model
from scripts.evaluate import evaluate_model

train_loader, test_loader, classes = load_data()

# Choose model
model = CNNModel()  # or MLPModel(), get_resnet18_model()

train_model(model, train_loader, test_loader, epochs=10)
evaluate_model(model, test_loader)
