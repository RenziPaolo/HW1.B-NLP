import json
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import Vocab, vocab
from collections import Counter
from git import Repo

class JSONLDataset(Dataset):
    """
    PyTorch dataset class constructed from a dictionary, ideally read from JSON Lines (JSONL) file.
    """

    def __init__(self, test: bool, device: str, tokenizer) -> None: 
        """
        Initialize the dataset with a dictionary.

        Args:
        test (bool): True if the dataset is for testing, False otherwise.
        """
        self.data_dict = self.read_json_file(test)
        self.device = device
        self.tokenizedData = []
        for sample in self.data_dict:
            text = sample['text']  # Assuming 'text' is the key for the text data
            tokenizedText = [token for token in tokenizer.tokenize(text)]  # Tokenizing text
            label = 0 if sample['label'] == 'si' else 1
            self.tokenizedData.append({'text': tokenizedText, 'label': label})
        self.indexedData = None
        #self.index(self.get_vocabulary())

    def read_json_file(self, test:  bool) -> list:
        """
        Read JSON data from a file and return it as a Python dictionary.

        Args:
        test (bool): True if the dataset is for testing, False otherwise.

        Returns:
        dict: The JSON data as a dictionary.
        """
        repo =  Repo(".", search_parent_directories=True)

        root_dir = repo.git.rev_parse("--show-toplevel")

        if test:
            file_path = root_dir+"/data/haspeede3-task1-test-data.jsonl"
        else:
            file_path = root_dir+"/data/haspeede3-task1-test-data.jsonl"
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data_list = [json.loads(line.strip()) for line in file]
                return data_list
        except FileNotFoundError:
            print("File not found.")
            return None
        except json.JSONDecodeError:
            print("Invalid JSON format.")
            return None

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.tokenizedData)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
        idx (int): Index of the sample.

        Returns:
        dict: Dictionary containing the sample data.
        """
        return self.tokenizedData[idx]
    
    def get_vocabulary(
        self,
        pad_token: str = "<pad>",
        unk_token: str = "<unk>",
        extra_tokens: list[str] = []
    ) -> Vocab:
        """Builds a `torchtext.vocab.Vocab` object from data stored in this object."""
        # most_common() returns a list of (token, count) pairs, so we convert them back into dictionary
        vocab_counter = dict(Counter(token for sent in self.tokenizedData for token in sent["text"]).most_common())
        # We build the vocabulary through a dictionary like {token: frequency, ...}
        vocabulary = vocab(vocab_counter, min_freq=1, specials=[pad_token, unk_token, *extra_tokens])
        # vocabulary(list of tokens) returns a list of values, so get the only one
        vocabulary.set_default_index(vocabulary([unk_token])[0])
        return vocabulary
    
    def index(self, vocabulary: Vocab) -> None:
        """Builds `self.indexedData` by converting raw samples to input_ids following `vocabulary`"""
        if self.indexedData is not None:
            print("Dataset has already been indexed. Keeping old index...")
        else:
            indexedData = []
            for sample in self.tokenizedData:
                # append the dictionary containing ids of the input tokens and label
                indexedData.append({"input_ids": vocabulary(sample["text"]), "label": sample["label"]})
            self.indexedData = indexedData
            print("indexed!")

    def indexBatch(self, vocabulary: Vocab, batch: list[dict]) -> None:
        """Builds `self.indexedData` by converting raw samples to input_ids following `vocabulary`"""
        if self.indexedData is not None:
            print("Dataset has already been indexed. Keeping old index...")
        else:
            indexedData = []
            for sample in batch:
                # append the dictionary containing ids of the input tokens and label
                indexedData.append({"input_ids": vocabulary(sample["text"]), "label": sample["label"]})
            return indexedData


    def _collate_fn(self, batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collate function to pad sequences within a batch.

        Args:
        batch (list): List of dictionaries, each containing a sample.

        Returns:
        dict: Dictionary containing padded sequences.
        """
        if "input_ids" not in batch[0].keys():
            batch = self.indexBatch(self.get_vocabulary(), batch)
        sequence_lengths = torch.tensor([len(sample["input_ids"]) for sample in batch], dtype=torch.long)
        labels = torch.tensor([sample['label'] for sample in batch])
        padded_texts = pad_sequence( (
                torch.tensor(sample["input_ids"], dtype=torch.long, device=self.device)
                for sample in batch
            ), batch_first=True)
        return  sequence_lengths, padded_texts, labels

    
# MAIN
if __name__ == '__main__' :
    from transformers import BertTokenizer
    dataset = JSONLDataset(test=True, device='cuda', tokenizer=BertTokenizer.from_pretrained("dbmdz/bert-base-italian-xxl-cased"))
    print(dataset[0])
