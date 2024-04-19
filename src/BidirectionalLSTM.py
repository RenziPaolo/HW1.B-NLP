import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

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
        embedding: bool = True
    ) -> None:

        super(BidirectionalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Embedding layer
        self.embedding = embedding
        if embedding:
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
        x (torch.Tensor): Input tensor of shape (sequence_lengths, input_ids).

        Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Get the different parts of the batch
        sequence_lengths, input_ids  = x
        if self.embedding:
            embeds = self.embedding(input_ids.to(self.device))
        packed = pack_padded_sequence(embeds, sequence_lengths, batch_first=True, enforce_sorted=False)

        # Initialize hidden state with zeros
        h0 = torch.rand(self.num_layers * 2, input_ids.size(0), self.hidden_size).to(self.device)
        # Initialize cell state with zeros
        c0 = torch.rand(self.num_layers * 2, input_ids.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        packed_output, (hidden_state, cell_state) = self.lstm(packed, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size * 2)
        
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        
        # Decode the hidden state of the last time step
        out = self.fc(hidden)  # out shape: (batch_size, output_size)

        return out
    
# MAIN
if __name__ == '__main__' :
    from dataset import JSONLDataset
    from transformers import BertTokenizer
    from torch.utils.data import DataLoader
    
    # Get the name of the GPU device
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    dataset = JSONLDataset(test=True, device=device, tokenizer=BertTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased"))
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=dataset._collate_fn)
    lstm = BidirectionalLSTM(len(dataset.get_vocabulary()), 128, 4, 0, 0, device)
    for step, (sequence_lengths, inputs, labels) in enumerate(dataloader):
        predictions = lstm((sequence_lengths, inputs))
        print(predictions)
        print(predictions.shape)
        quit()