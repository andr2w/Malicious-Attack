import re
from torch.utils.data import Dataset

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




def FakeDataset():
    def __init__(self, X, y, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.X = X
        self.y = y
        self.max_len = max_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = str(self.X[index])
        X = " ".join(X.split())

        inputs = self.tokenizer.encode_plus(
            X,
            None, 
            add_special_tokens = True,
            max_length = self.max_len,
            pad_to_max_length = True,
            return_token_type_ids = True
        )

        return inputs


def build_vocab(data_path, tokenizer, max_len):
    X, y  = read_data(data_path)
    return FakeDataset(X, y, tokenizer, max_len) 