from re import I
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class Config:
    def __init__(self):
        self.bert_model = 'bert-base-uncased'
        self.data_path = 'Fake'
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
        self.saved_path = 'model.ckpt' 
'''
According to the GitHub/google-research/bert repo

Model: 'bert-base-uncased' is a 
-  12-layer
-  12-heads
- **Output-dims** is 768 ==> hidden_size = 768
''' 
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_model)
        # We do not want to update the bert's parameters
        # We do not want to compute the gradient of the bert's parameters
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # The output of the bert model is 768
        self.fc1 = nn.Linear(config.bert_output_size, config.fc1_output_size)
        self.dropout = nn.Dropout(config.drop_out)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(config.fc1_output_size, config.num_classes)
    
    def forward(self, input_ids, attention_mask):
        '''
        The output for bert layer is a list contains two elements;
        1. encoder_out, i.e., out[0].
        2. pooled_out, i.e., out[1].
        
        They are both contains 784 output dims,

        Well actually;

        Encoder-786-dim
             |
             |
        Pooled_out-786-dim

        We will use Pooled_out. 
        You can use Encoder out if you want.
        '''
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_out = out[1]
        out = self.fc1(pooled_out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out