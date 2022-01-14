import re
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
import torch
from sklearn.model_selection import train_test_split

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
        '''
        Padding = ['longest', 'max_length', 'do_not_pad']
        '''
        inputs = self.tokenizer.encode_plus(
            X,
            None, 
            add_special_tokens = True,
            padding = 'max_length',
            return_token_type_ids = True,
            truncation = True # default false, cut the length of the text
        )
        '''
        The Var inputs, is basically a `dict`.
        - 1. Input_ids, the index of the word in the Bert model.
        - 2. Token_type_ids, The BERT model is to do classifcitation on pairs of sentences, i.e., document level. However, in this case all should be 1. 
        - 3. attention_mask, The attention mask is a binary indicating the position of the padded indices so that the model does not attend to them, e.g., if it is a real word, then the model will not attend to them.
        '''
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.LongTensor(ids),
            'mask': torch.LongTensor(mask),
            'y': torch.LongTensor([y])
        }


def build_vocab(data_path, tokenizer, train_size):
    '''
    To do:
    1. Split the dataset: train, test, validation.
    2. Init the Fake dataset.
    3. Build the data Iterator. # maybe not, build the iter in the next function
    '''
    X, y = read_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - train_size, shuffle=True, random_state=333)

    print('Full Dataset:', len(X))
    print('Train Dataset:', len(X_train))
    print('Test Dataset:', len(X_test))

    train_set = FakeDataset(X_train, y_train, tokenizer)
    test_set = FakeDataset(X_test, y_test, tokenizer)

    
    return train_set, test_set




def build_iter(train_set, test_set, batch_size):
    train_iter = DataLoader(train_set,
                            batch_size = batch_size, 
                            shuffle = True,
                            num_workers = 4)

    test_iter = DataLoader(test_set,
                           batch_size = batch_size,
                           shuffle = True,
                           num_workers = 4)
    
    return train_iter, test_iter
