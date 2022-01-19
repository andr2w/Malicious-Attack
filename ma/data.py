import re
from torch.utils.data import Dataset, DataLoader
import torch


def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()


def read_data(path):    
    features = []
    labels = []
    with open(path, encoding='utf8') as f:
        for line in f:
            y, _, X = line.partition(' ')
            y = int(y)
            
            # clean the text
            X = clean_str(X.strip())            
            # lower the word 
            X = X.lower() 
            
            features.append(X)
            labels.append(y)
        

    return features, labels    


class FakeDataset(Dataset):
    def __init__(self, X, y, tokenizer):
        self.X = X
        self.y = y
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        y = self.y[index]

        inputs = self.tokenizer.encode_plus(
            X, 
            None,
            add_special_tokens = True,
            padding = 'max_length',
            return_token_type_ids = True,
            truncation = True
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'y': torch.LongTensor([y])
        }

def build_vocab(data_path, tokenizer):
    X, y = read_data(data_path)
    data_set = FakeDataset(X, y, tokenizer)
    return data_set


def build_iter(data_set, batch_size):
    data_iter = DataLoader(data_set, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_iter