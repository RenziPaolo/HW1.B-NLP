import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.trainer import Trainer
from src.BidirectionalLSTM import BidirectionalLSTM
from src.BiLSTMModel import BiLSTMModel
from src.dataset import JSONLDataset
import spacy


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
dataset = JSONLDataset(test=True, device=device, tokenizer=spacy.load("it_core_news_sm"))
# Define the sizes of training, validation, and test sets
train_size = int(0.9 * len(dataset))  # 90% of the data for training
val_size = int(0.1 * len(dataset))   # 10% of the data for validation
batch_size = 128

vocabulary = dataset.get_vocabulary()

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])



# Create DataLoader instances for each set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset._collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=dataset._collate_fn)
lstm = BidirectionalLSTM(len(vocabulary), 1, 1, 0.5, 2, vocabulary(["<pad>"])[0], device=device)

trainer = Trainer(
model=lstm,
optimizer=torch.optim.Adam(lstm.parameters(), lr=0.0001),
#loss_function=nn.CrossEntropyLoss(),
log_steps=100
)

losses = trainer.train(train_loader, val_loader, epochs=10)

