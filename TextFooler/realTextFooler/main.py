from model import Config, Model
from data import build_vocab, build_iter
import torch
from trainer import train, test



def main():
    config = Config()
    train_set, test_set = build_vocab(config.data_path, config.tokenizer, config.train_size)
    train_iter, test_iter = build_iter(train_set, test_set, config.batch_size)
    net = Model(config).to(device=config.device)
    optimizer = torch.optim.Adam(params= net.parameters(), lr=config.learning_rate) 
    loss_function = torch.nn.CrossEntropyLoss()
    
    train(train_iter, net, optimizer, loss_function, config, debug=False)  
    test(test_iter, net, config)


if __name__ == '__main__':
    main()