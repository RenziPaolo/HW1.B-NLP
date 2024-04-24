from src.baselines import RandomBase
from src.trainer import Trainer
import torch
from src.dataset import JSONLDataset
import spacy
from torch.utils.data import DataLoader

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

dataset = JSONLDataset(test=False, device=device, tokenizer=spacy.load("it_core_news_sm"))
dataloader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=dataset._collate_fn)
trainer = Trainer(
model=RandomBase(),
optimizer=None,
log_steps=100
)
test_loss, test_acc = trainer.evaluate(dataloader)

print(f"Random Baseline: Test loss {test_loss}, Test accuracy: {test_acc}")
