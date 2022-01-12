import torch.nn as nn
import torch

class Config:
    def __init__(self):
        self.data_path = 'Pheme.csv'
        self.batch_size = 64
        self.min_freq = 0
        self.pad_length = 80
        self.embedding_path = '../../Glove_Twitter_wordVec/glove.twitter.27B.200d.txt'
        self.embed_size = 200
        self.num_hiddens = 128
        self.num_layers = 2
        self.num_classes = 2
        self.num_epochs = 10
        self.learning_rate = 1e-5
        self.saved_path = 'model2.ckpt'

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
        self.decoder = nn.Linear(num_hiddens * 4, num_classes) # 2 * 2 == 4

    def forward(self, x):
        embeddings = self.embedding(x.T)
        # for faster storing
        self.encoder.flatten_parameters()
        out, _ = self.encoder(embeddings)
        # Only get the latest hidden state
        out = self.decoder(torch.cat((out[0], out[-1]), dim = 1))
        return out


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
    if type(m) == nn.LSTM:
        for param in m._flat_weights_names:
            if "weight" in param:
                nn.init.xavier_uniform_(m._parameters[param])
       
        

