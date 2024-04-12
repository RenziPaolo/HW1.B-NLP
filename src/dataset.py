import json
from torch.utils.data import Dataset

def read_json_file(test):
    """
    Read JSON data from a file and return it as a Python dictionary.
    
    Args:
    file_path (str): The path to the JSON file.
    
    Returns:
    dict: The JSON data as a dictionary.
    """
    if test:
        file_path = "../data/haspeede3-task1-test-data.jsonl"
    else:
        file_path = "../data/haspeede3-task1-test-data.jsonl"
    try:
        with open(file_path, 'r') as file:
            data_list = [json.loads(line.strip()) for line in file]
            return data_list
    except FileNotFoundError:
        print("File not found.")
        return None
    except json.JSONDecodeError:
        print("Invalid JSON format.")
        return None
    
class JSONLDataset(Dataset):
    """
    PyTorch dataset class constructed from a dictionary, ideally read from JSON Lines (JSONL) file.
    """

    def __init__(self, data_dict):
        """
        Initialize the dataset with a dictionary.

        Args:
        data_dict (dict): Dictionary containing the data.
        """
        self.data_dict = data_dict

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.data_dict)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
        idx (int): Index of the sample.

        Returns:
        dict: Dictionary containing the sample data.
        """
        return self.data_dict[idx]
    
# MAIN
if __name__ == '__main__' :
    test_json = read_json_file(True)
    dataset = JSONLDataset(test_json)
    print(dataset[0])