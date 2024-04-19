import torch
from torch.utils.data import DataLoader, random_split
from src.trainer import Trainer
from src.model import BidirectionalLSTM
from src.dataset import JSONLDataset
from transformers import BertTokenizer

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

dataset = JSONLDataset(test=True, device=device, tokenizer=BertTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased"))
# Define the sizes of training, validation, and test sets
train_size = int(0.9 * len(dataset))  # 90% of the data for training
val_size = int(0.1 * len(dataset))   # 10% of the data for validation
batch_size = 128

# Split the dataset into training, validation, and test sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoader instances for each set
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset._collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=dataset._collate_fn)

lstm = BidirectionalLSTM(len(dataset.get_vocabulary()), 128, 4, 0, 0, device)
trainer = Trainer(
model=lstm,
optimizer=torch.optim.Adam(lstm.parameters(), lr=0.0001),
log_steps=100
)
losses = trainer.train(train_loader, val_loader, epochs=10)
print(losses)


