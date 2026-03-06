from torch import nn
import torch

class AuctionModel(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, x, mask=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        if mask is not None:
            # Mask before softmax: illegal actions get -inf so they get 0 prob
            x = x.masked_fill(mask == 0, float("-inf"))
        x = torch.softmax(x, dim=1)
        return x