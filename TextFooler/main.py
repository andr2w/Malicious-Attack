from model import Config, BiLstm, init_weights
from data import TokenEmbedding, return_data, Vocab, build_iterator
import torch 
import torch.nn as nn
# from trainer import train, test
from trainer import train



def main(): 
    config = Config()
    # X_train, X_test, y_train, y_test = return_data(config.data_path)
    X_train, y_train = return_data(config.data_path)
    vocab = Vocab(X_train, min_freq=config.min_freq, reserved_tokens=['<pad>'])
    print(len(vocab))
    pretrain_embedding = TokenEmbedding(config.embedding_path)
    embeds = pretrain_embedding[vocab.idx_to_token]
    print(embeds.shape)
    # train_iter, test_iter = build_iterator(X_train, X_test, y_train, y_test, vocab, config) 
    train_iter = build_iterator(X_train, y_train, vocab, config)   

    net = BiLstm(len(vocab), config.embed_size, config.num_hiddens, config.num_layers, config.num_classes)
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False

    net.apply(init_weights)

    optimizer = torch.optim.Adam(net.parameters(), lr=config.learning_rate)
    loss_function = nn.CrossEntropyLoss()
    
    train(train_iter, net, optimizer, loss_function, config)
    # test(test_iter, net, config)




if __name__ == '__main__':
    main()

