import torch
import matplotlib.pyplot as plt
from model import NuraleNet
from dataset import MNISTDataset


# Hyperparams
dataset_path = torch.load("./data/MNIST/processed/mnist_test.pt")
saved_model = torch.load('./pytorch/trained_model.pth')
device = device = "cuda" if torch.cuda.is_available() else "cpu"
# ----------


def drawplt(model, data, num_plots=30):
    x, y = data
    x, y = x.to(device), y.to(device)
    preds = model(x).argmax(axis=1)

    fig, axes = plt.subplots(num_plots//10, 10, figsize=(20, 15))
    fig.suptitle("Predictions")
    for i, ax in enumerate(axes.flat):
        ax.imshow(x[i])
        ax.set_title(f'Prediction: {preds[i]}')
        ax.axis('off')

    plt.show()


if __name__ == "__main__":
    model = NuraleNet()
    model.load_state_dict(saved_model['model_state_dict'])

    test = MNISTDataset(dataset_path)

    drawplt(model, test[0:30])
