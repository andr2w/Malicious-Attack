import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_avaiable() else 'cpu')


embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000
batch_size = 20 
seq_length = 30
learning_rate = 0.002
