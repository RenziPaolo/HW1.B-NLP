import torch
import torch.nn as nn
from dataset import JSONLDataset
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM model.
    """

    def __init__(
        self,
        vocabulary_length: int,
        hidden_size: int,
        num_layers: int,
        bilstm_dropout: float,
        padding_id: int,
        device: str = "cuda",
    ) -> None:

        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_length,
            embedding_dim=hidden_size,
            padding_idx=padding_id, # avoid updating the gradient of padding entries
            device=device
        )
        self.device = device
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=bilstm_dropout, device=device)

        # Output layer
        self.fc = nn.Linear(hidden_size * 2, 2, device=device)

    def forward(self, x: list[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the Bidirectional LSTM model.

        Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        print(x[1].shape)
        # Get the different parts of the batch
        input_ids, sequence_lengths = x

        embeds = self.embedding(input_ids.to(self.device))
        print(embeds.shape)
        packed = pack_padded_sequence(embeds, sequence_lengths, batch_first=True, enforce_sorted=False)

        # Initialize hidden state with zeros
        h0 = torch.rand(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state with zeros
        c0 = torch.rand(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(packed, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size * 2)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])  # out shape: (batch_size, output_size)

        return out
    
# MAIN
if __name__ == '__main__' :
    dataset = JSONLDataset(test=True, device='cuda')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=dataset._collate_fn)
    lstm = BidirectionalLSTM(len(dataset.get_vocabulary()), 128, 4, 0, 0, 'cuda')
    for step, (sequence_lengths, inputs, labels) in enumerate(dataloader):

        predictions = lstm((sequence_lengths, inputs))