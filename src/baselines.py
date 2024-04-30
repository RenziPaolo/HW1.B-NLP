import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.utils.rnn as rnn_utils

class RandomBase(nn.Module):

    def __init__(self, device):
        super(RandomBase, self).__init__()
        self.device = device
    
    def forward(self, x: list[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return torch.rand([x[0].size(0), 2], device=self.device)

class RNN(nn.Module):
    def __init__(self, hidden_size, output_size, vocabulary_length, padding_id, device):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_length,
            embedding_dim=hidden_size[0],
            padding_idx=padding_id, # avoid updating the gradient of padding entries
            device=device
        )
        self.rnn = nn.RNN(hidden_size[0], hidden_size[1], batch_first=True, device=device)
        self.fc = nn.Linear(hidden_size[1], out_features=output_size, device=device)

    def forward(self, x):
        # Forward pass through the RNN layer
        sequence_lengths, input_ids = x

        # First we embed the input tokens
        embeds = self.embedding(input_ids) # [B, S, H]

        packed = pack_padded_sequence(embeds, sequence_lengths, batch_first=True, enforce_sorted=False)

        out, hidden_state = self.rnn(packed)

        padded_sequence, lengths = rnn_utils.pad_packed_sequence(out, batch_first=True)
        # Passing the output through the fully connected layer
        out = self.fc(padded_sequence[:, 0, :])
        

        return out
