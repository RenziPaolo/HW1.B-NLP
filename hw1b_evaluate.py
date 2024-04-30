from src.baselines import RandomBase
from src.trainer import Trainer
import torch
from src.dataset import JSONLDataset
import spacy
from torch.utils.data import DataLoader
from src.BidirectionalLSTM import BidirectionalLSTM
from src.baselines import RNN
import os
import glob

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
dataset = JSONLDataset(test=False, device=device, tokenizer=spacy.load("it_core_news_sm"))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=dataset._collate_fn)
trainer = Trainer(
model=RandomBase(device),
optimizer=None,
log_steps=100
)
test_loss, test_acc = trainer.evaluate(dataloader)

print(f"Random Baseline: Test loss {test_loss}, Test accuracy: {test_acc}")

vocabulary = dataset.get_vocabulary()

lstm = BidirectionalLSTM(len(vocabulary), 1, 1, 0.5, 2, vocabulary(["<pad>"])[0], device=device)
rnn = RNN( (1,1), 2, len(vocabulary), vocabulary(["<pad>"])[0], device=device)

# Directory where your files are stored
directory = './saves/'

# Define the pattern to search for
pattern = 'RNN*'

# Get a list of all files matching the pattern
files = glob.glob(os.path.join(directory, pattern))

# Filter the files based on the format "lstmYYYY-MM-DD"
files_with_date = [file for file in files if file.startswith(directory+'RNN')]

print("files_with_date:",files_with_date)

# Sort the files based on the date in descending order
files_with_date.sort(reverse=True)

# Get the latest file path
checkpoint_path_rnn = files_with_date[0] if files_with_date else None

rnn_checkpoint = torch.load(checkpoint_path_rnn)

rnn.load_state_dict(rnn_checkpoint)

rnn.eval()

trainer = Trainer(
model=rnn,
optimizer=None,
log_steps=100
)

test_loss, test_acc = trainer.evaluate(dataloader)

print(f"rnn: Test loss {test_loss}, Test accuracy: {test_acc}")


# Define the pattern to search for
pattern = 'lstm*'

# Get a list of all files matching the pattern
files = glob.glob(os.path.join(directory, pattern))

# Filter the files based on the format "lstmYYYY-MM-DD"
files_with_date = [file for file in files if file.startswith(directory+'lstm')]

# Sort the files based on the date in descending order
files_with_date.sort(reverse=True)

# Get the latest file path
checkpoint_path_lstm = files_with_date[0] if files_with_date else None

lstm_checkpoint = torch.load(checkpoint_path_lstm)

lstm.load_state_dict(lstm_checkpoint)

lstm.eval()

trainer = Trainer(
model=lstm,
optimizer=None,
log_steps=100
)

test_loss, test_acc = trainer.evaluate(dataloader)

print(f"lstm: Test loss {test_loss}, Test accuracy: {test_acc}")