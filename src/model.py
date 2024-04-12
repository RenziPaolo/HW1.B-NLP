import torch
import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM model.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        """
        Initialize the Bidirectional LSTM model.

        Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int): Number of recurrent layers.
        output_size (int): The number of output features.
        """
        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, device=device)

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, output_size, device=device)

    def forward(self, x):
        """
        Forward pass of the Bidirectional LSTM model.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_length, input_size).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Initialize hidden state with zeros
        h0 = torch.rand(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state with zeros
        c0 = torch.rand(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size * 2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # out shape: (batch_size, output_size)

        return out