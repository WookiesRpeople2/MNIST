import torch
import torch.nn as nn
import torch.nn.init as init


class NuraleNet(nn.Module):
    def __init__(self):
        super().__init__()
        # beacause x.shape = 60000, 28, 28 we make the stating layer 28 to the power of 2
        self.ln1 = nn.Linear(28**2, 200)
        self.ln2 = nn.Linear(200, 50)
        self.ln3 = nn.Linear(50, 10)

        self.bn1 = nn.LayerNorm(200)
        self.bn2 = nn.LayerNorm(50)
        self.bn3 = nn.LayerNorm(10)

        self.relu = nn.ReLU()

        init.kaiming_uniform_(self.ln1.weight)
        init.kaiming_uniform_(self.ln2.weight)
        init.kaiming_uniform_(self.ln3.weight)

    def forward(self, x):
        x = x.view(-1, 28**2)
        x = self.ln1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.ln2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.ln3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x.squeeze()
