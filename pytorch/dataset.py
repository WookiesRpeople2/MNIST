from torch.utils.data import Dataset
import torch.nn.functional as F


class MNISTDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.x, self.y = dataset.data, dataset.targets
        # we want values in between 0 and 1, 255 is the max for colors 0-255
        self.x = self.x / 255
        self.y = F.one_hot(self.y, num_classes=10).to(float)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]
