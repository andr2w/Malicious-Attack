import re
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

def clean_text(text):
    """
    This function cleans the text in the following ways
    1. Replace websites with URL
    2. Replace 's with <space>'s (e.g., her's --> her 's)
    """
    text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "URL", text) # Replace urls with special token
    #text = text.replace("\'s", "")
    #text = text.replace("\'", "")
    #text = text.replace("n\'t", " n\'t")
    text = text.replace("@", "")
    text = text.replace(":", "")
    text = text.replace("#", "")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("&amp;", "")
    text = text.replace("&gt;", "")
    text = text.replace("\"", "")
    text = text.replace("$MENTION$", '')
    text = text.replace("$ URL $", '')
    text = text.replace("$URL$", '')
    text = text.replace(".", "")
    text = text.replace(",", "")
    text = text.replace("(", "")
    text = text.replace(")", "")
    text = text.replace("<end>", "")
    text = text.replace("|", "")
    text = text.lower()
    return text.strip()

def change_label(label):
    if label == 'non-rumours':
        label = 0
    elif label == 'rumours':
        label = 1
    
    return label

def return_data(data_path):
    dataset = pd.read_csv(data_path)
    X = dataset['Text']
    y = dataset['Label']

    X = X.apply(lambda x: clean_text(x))
    X = X.apply(lambda x: x.split())
    y = y.apply(lambda x: change_label(x))

    '''
    The Train_test_split don't work as i wished, the y label is not correct
    Let me first do some test.
    '''
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100, shuffle=True)

    return X_train, X_test, y_train, y_test



class Config:
    def __init__(self):
        self.bert_model = 'bert-base-uncased'
        self.data_path = 'Pheme.csv'
        self.train_size = 0.8
        self.num_epochs = 15
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model)
        self.batch_size = 32
        self.bert_output_size = 768
        self.fc1_output_size = 268
        self.num_classes = 2
        self.drop_out = 0.2
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = 5e-5
        self.saved_path = 'model_PHE.ckpt' 



class PhemeDataset(Dataset):
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


def build_vocab(X_train, X_test, y_train, y_test, tokenizer):

    train_set = PhemeDataset(X_train, y_train, tokenizer)
    test_set = PhemeDataset(X_test, y_test, tokenizer)

    
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