from re import I
import torch
import torch.nn as nn
import torch.nn.functional as f


class Config:
    def __init__(self):
        self.data_path = 'Pheme.csv'
        self.batch_size = 64
        self.min_freq = 3
        self.pad_length = 35
        self.embedding_path = '../../glove.6B.100d/vec.txt'


        '''
        embedding:
        Input: (vocab_size, embed_size), i.e., len(embeds), glove_vecs_dim
        Output: (batch_size, padding_length, word_dims)
        '''
class BiLstm(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, num_classes):
        super(BiLstm, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.encoder = nn.LSTM(embed_size, num_hiddens, num_layers=num_layers, bidirectional=True)
        self.decoder = nn.Linear(num_hiddens * 2, num_classes) # 2 * 2 == 4

    def forward(self, x):
        embeddings = self.embedding(x.T)
        # for faster storing
        self.encoder.flatten_parameters()
        out, _ = self.encoder(embeddings)
        # Only get the latest hidden state
        out = self.decoder(out[-1])
        return out




       
        

