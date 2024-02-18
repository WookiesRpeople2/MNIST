import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import NuraleNet
from dataset import MNISTDataset


# HyperParams
n_epochs = 30
lr = 0.01
batch_size = 8
device = device = "cuda" if torch.cuda.is_available() else "cpu"
dataset_path = torch.load("./data/MNIST/processed/mnist_train.pt")
model_save_path = "./pytorch/trained_model.pth"
# --------------------

if __name__ == "__main__":

    train = MNISTDataset(dataset_path)
    train_dl = DataLoader(train, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    model = NuraleNet()

    # train the model
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(n_epochs):
        print(epoch)
        size = len(train_dl)
        for i, (x, y) in enumerate(train_dl):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y).to(device)
            loss.backward()
            optimizer.step()

    torch.save({
        'epoch': n_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_save_path)
