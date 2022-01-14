from model_phe import return_data, Config, build_vocab, build_iter
from model import Model
import torch
from trainer import train, test


def main():
    config = Config()
    X_train, X_test, y_train, y_test = return_data(config.data_path) 
    print('Train Dataset:', len(X_train))
    print('Test Dataset:', len(X_test))

    train_set, test_set = build_vocab(X_train, X_test, y_train, y_test, config.tokenizer)
    train_iter, test_iter = build_iter(train_set, test_set, config.batch_size)
    
    net = Model(config).to(device=config.device)
    optimizer = torch.optim.Adam(params= net.parameters(), lr=config.learning_rate) 
    loss_function = torch.nn.CrossEntropyLoss()

    train(train_iter, net, optimizer, loss_function, config, debug=False) 
    test(test_iter, net, config)


if __name__ == '__main__':
    main() 