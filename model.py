from torch import nn
import torch


class AuctionModel(nn.Module):
    """MLP policy for auction bidding."""

    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        # Bias against action 0 (drop): encourages bidding so RL agent wins more
        with torch.no_grad():
            self.fc3.bias[0] = -2.0

    def forward(self, x, mask=None, hidden=None):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        if mask is not None:
            # Mask before softmax: illegal actions get -inf so they get 0 prob
            x = x.masked_fill(mask == 0, float("-inf"))
        x = torch.softmax(x, dim=1)
        return x


class AuctionLSTMModel(nn.Module):
    """LSTM policy for auction bidding; maintains hidden state across steps in an episode."""

    def __init__(self, input_size, hidden_size, action_size, lstm_hidden_size=64):
        super().__init__()
        self.lstm_hidden_size = lstm_hidden_size
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True)
        self.fc_out = nn.Linear(lstm_hidden_size, action_size)
        with torch.no_grad():
            self.fc_out.bias[0] = -2.0

    def forward(self, x, mask=None, hidden=None):
        # x: (batch, input_size)
        x = torch.relu(self.fc_in(x))
        x = x.unsqueeze(1)  # (batch, 1, hidden_size)
        if hidden is None:
            out, new_hidden = self.lstm(x)
        else:
            out, new_hidden = self.lstm(x, hidden)
        out = out.squeeze(1)  # (batch, lstm_hidden_size)
        logits = self.fc_out(out)
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float("-inf"))
        probs = torch.softmax(logits, dim=1)
        return probs, new_hidden